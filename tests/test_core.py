import pytest
from hwtypes import BitVector
import magma as m
import fault as f
from mantle import CounterModM, RegFileBuilder

from riscv_mini.core import Core
from .utils import concat


class SimpleTests:
    tests = ["rv32ui-p-simple"]
    maxcycles = 150000


class ISATests:
    tests = []
    for test in ["simple", "add", "addi", "auipc", "and", "andi",
                 "sb", "sh", "sw", "lb", "lbu", "lh", "lhu", "lui", "lw",
                 "beq", "bge", "bgeu", "blt", "bltu", "bne", "j", "jal",
                 "jalr", "or", "ori", "sll", "slli", "slt", "slti", "sra",
                 "srai", "sub", "xor", "xori"]:
        # TODO: "fence_i" (also todo in chisel riscv-mini)
        tests.append(f"rv32ui-p-{test}")
    for test in ["sbreak", "scall", "illegal", "ma_fetch", "ma_addr", "csr"]:
        # TODO: "timer" (also todo in chisel riscv-mini)
        tests.append(f"rv32mi-p-{test}")
        maxcycles = 15000


class BmarkTests:
    tests = ["median.riscv", "multiply.riscv", "qsort.riscv", "towers.riscv",
             "vvadd.riscv"]
    maxcycles = 1500000


class LargeBmarkTests:
    tests = ["median.riscv-large", "multiply.riscv-large", "qsort.riscv-large",
             "towers.riscv-large", "vvadd.riscv-large"]
    maxcycles = 5000000


def parse_nibble(hex_str):
    hex_val = ord(hex_str)
    if hex_val >= ord('a'):
        return hex_val - ord('a') + 10
    return hex_val - ord('0')


def load_mem(lines, chunk):
    insts = []
    for line in lines:
        assert len(line) % (chunk // 4) == 0
        for i in range(len(line) - (chunk // 4), -1, -(chunk // 4)):
            inst = 0
            for j in range(0, chunk // 4):
                inst |= (parse_nibble(line[i + j]) <<
                         (4 * ((chunk // 4) - (j + 1))))
            insts.append(BitVector[chunk](inst))
    return insts


@pytest.mark.parametrize('test', [SimpleTests, ISATests, BmarkTests])
def test_core(test):
    for t in test.tests:
        x_len = 32
        with open(f'tests/resources/{t}.hex', 'r') as file_:
            contents = [line.rstrip() for line in file_.readlines()]
            loadmem = load_mem(contents, x_len)

        class DUT(m.Circuit):
            io = m.IO(done=m.Out(m.Bit)) + m.ClockIO(has_reset=True)
            core = Core(x_len)()
            core.host.fromhost.data.undriven()
            core.host.fromhost.valid @= False

            # reverse concat because we're using utils with chisel ordering
            _hex = [concat(*reversed(x)) for x in loadmem]
            imem = RegFileBuilder("imem", 1 << 20, x_len, write_forward=False,
                                  reset_type=m.Reset, backend="verilog")
            dmem = RegFileBuilder("dmem", 1 << 20, x_len, write_forward=False,
                                  reset_type=m.Reset, backend="verilog")

            INIT, RUN = False, True

            state = m.Register(init=INIT)()
            cycle = m.Register(m.UInt[32])()

            n = len(_hex)
            counter = CounterModM(n, n.bit_length(), has_ce=True)
            counter.CE @= m.enable(state.O == INIT)
            cntr, done = counter.O, counter.COUT

            iaddr = (core.icache.req.data.addr // (x_len // 8))[:20]
            daddr = (core.dcache.req.data.addr // (x_len // 8))[:20]

            dmem_data = dmem[daddr]
            imem_data = imem[iaddr]
            write = 0
            for i in range(x_len // 8):
                write |= m.zext_to(m.mux(
                    [dmem_data, core.dcache.req.data.data],
                    core.dcache.req.valid & core.dcache.req.data.mask[i]
                )[8 * i:8 * (i + 1)], 32) << (8 * i)

            core.RESET @= m.reset(state.O == INIT)

            core.icache.resp.valid @= state.O == RUN
            core.dcache.resp.valid @= state.O == RUN

            core.icache.resp.data.data @= m.Register(m.UInt[x_len])()(imem_data)
            core.dcache.resp.data.data @= m.Register(m.UInt[x_len])()(dmem_data)

            chunk = m.mux(_hex, cntr)

            imem.write(m.zext_to(cntr, 20), chunk, m.enable(state.O == INIT))

            dmem.write(
                m.mux([m.zext_to(cntr, 20), daddr], state.O == INIT),
                m.mux([chunk, write], state.O == INIT),
                m.enable(
                    (state.O == INIT) | (core.dcache.req.valid &
                                         core.dcache.req.data.mask.reduce_or()))
            )

            @m.inline_combinational()
            def logic():
                state.I @= state.O
                cycle.I @= cycle.O
                if state.O == INIT:
                    if done:
                        state.I @= RUN
                if state.O == RUN:
                    cycle.I @= cycle.O + 1

            m.display("LOADMEM[%x] <= %x", cntr * (x_len // 8),
                      chunk).when(m.posedge(io.CLK)).if_(state.O == INIT)

            m.display("INST[%x] => %x", iaddr * (x_len // 8),
                      dmem_data).when(m.posedge(io.CLK)).if_(
                          (state.O == RUN) & core.icache.req.valid)

            m.display("MEM[%x] <= %x", daddr * (x_len // 8),
                      write).when(m.posedge(io.CLK)).if_(
                          (state.O == RUN) & core.dcache.req.valid &
                          core.dcache.req.data.mask.reduce_or())

            m.display("MEM[%x] => %x", daddr * (x_len // 8),
                      dmem_data).when(m.posedge(io.CLK)).if_(
                          (state.O == RUN) & core.dcache.req.valid &
                          ~core.dcache.req.data.mask.reduce_or())
            f.assert_immediate(cycle.O < test.maxcycles)
            io.done @= core.host.tohost != 0
            f.assert_immediate((core.host.tohost >> 1) == 0,
                              failure_msg=("* tohost: %d *", core.host.tohost))

            m.display("cycles: %d", cycle.O).when(m.posedge(io.CLK)).if_(
                io.done.value() == 1)

    tester = f.Tester(DUT, DUT.CLK)
    tester.wait_until_high(DUT.done)
    tester.compile_and_run("verilator", magma_opts={"inline": True,
                                                    "verilator_compat": True},
                           flags=['-Wno-unused', '-Wno-undriven', '--assert'],
                           disp_type="realtime")
