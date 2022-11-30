import pytest
import fault as f
import magma as m
from mantle import RegFileBuilder
import mantle2

from riscv_mini.data_path import Datapath, Const
from riscv_mini.control import Control
from riscv_mini.imm_gen import ImmGenWire, ImmGenMux
from .utils import tests, test_results, fin, nop


@pytest.mark.parametrize('test', ['bypass', 'exception'])
@pytest.mark.parametrize('ImmGen', [ImmGenWire, ImmGenMux])
def test_datapath(test, ImmGen):
    class DUT(m.Circuit):
        x_len = 32
        io = m.IO(done=m.Out(m.Bit)) + m.ClockIO(has_reset=True)
        data_path = Datapath(x_len, ImmGen=ImmGen)()
        control = Control(x_len)()
        for name, value in data_path.ctrl.items():
            m.wire(value, getattr(control, name))
        data_path.host.fromhost.data.undriven()
        data_path.host.fromhost.valid @= 0

        insts = tests[test]
        INIT, RUN = False, True
        state = m.Register(init=INIT)()
        n = len(insts)
        counter = mantle2.CounterTo(n, has_enable=True)()
        counter.CE @= m.enable(state.O == INIT)
        cntr, done = counter.O, counter.COUT
        timeout = m.Register(m.Bits[x_len])()
        mem = RegFileBuilder("mem", 1 << 20, x_len, write_forward=False,
                             reset_type=m.Reset, backend="verilog")
        iaddr = (data_path.icache.req.data.addr // (x_len // 8))[:20]
        daddr = (data_path.dcache.req.data.addr // (x_len // 8))[:20]
        write = 0
        mem_daddr = mem[daddr]
        mem_iaddr = mem[iaddr]
        for i in range(x_len // 8):
            write |= m.mux([
                mem_daddr & (0xff << (8 * i)),
                data_path.dcache.req.data.data
            ], data_path.dcache.req.data.mask[i])
        data_path.RESET @= m.reset(state.O == INIT)
        data_path.icache.resp.data.data @= \
            m.Register(m.UInt[x_len])()(mem_iaddr)
        data_path.icache.resp.valid @= state.O == RUN

        data_path.dcache.resp.data.data @= \
            m.Register(m.UInt[x_len])()(mem_daddr)
        data_path.dcache.resp.valid @= state.O == RUN

        for addr in range(0, Const.PC_START, 4):
            wdata = fin if addr == Const.PC_EVEC + (3 << 6) else nop
            mem.write(addr // 4, wdata, m.enable(state.O == INIT))
        mem.write(Const.PC_START // (x_len // 8) + m.zext_to(cntr, 20),
                  m.mux(insts, cntr),
                  m.enable(state.O == INIT))

        mem.write(daddr, write,
                  m.enable((state.O == RUN) & data_path.dcache.req.valid &
                           data_path.dcache.req.data.mask.reduce_or()))

        m.display("INST[%x] = %x, iaddr: %x", data_path.icache.req.data.addr,
                  mem_iaddr, iaddr).when(m.posedge(io.CLK))\
            .if_((state.O == RUN) & data_path.icache.req.valid)

        m.display("MEM[%x] <= %x", data_path.dcache.req.data.addr,
                  write).when(m.posedge(io.CLK))\
            .if_((state.O == RUN) & data_path.dcache.req.valid &
                 data_path.dcache.req.data.mask.reduce_or())

        m.display("MEM[%x] => %x", data_path.dcache.req.data.addr,
                  mem_daddr).when(m.posedge(io.CLK))\
            .if_((state.O == RUN) & data_path.dcache.req.valid &
                 ~data_path.dcache.req.data.mask.reduce_or())

        @m.inline_combinational()
        def logic():
            state.I @= state.O
            timeout.I @= timeout.O
            io.done @= False
            if state.O == INIT:
                if done:
                    state.I @= RUN
            elif state.O == RUN:
                timeout.I @= timeout.O + 1
                if data_path.host.tohost != 0:
                    io.done @= True

        f.assert_immediate(
            (state.O != RUN) | (data_path.host.tohost == 0) |
            (data_path.host.tohost == test_results[test]),
            failure_msg=(f"* tohost: %d != {test_results[test]} *",
                         data_path.host.tohost)
        )

    tester = f.Tester(DUT, DUT.CLK)
    tester.wait_until_high(DUT.done)
    tester.compile_and_run("verilator", magma_opts={"inline": True,
                                                    "verilator_compat": True},
                           flags=['-Wno-unused', '-Wno-undriven', '--assert'],
                           disp_type="realtime")
