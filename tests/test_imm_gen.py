import magma as m
import mantle
import fault as f

from riscv_mini.control import (Control, IMM_I, IMM_S, IMM_B, IMM_U, IMM_J,
                                IMM_Z)
from riscv_mini.imm_gen import ImmGenWire

from .utils import insts, iimm, simm, bimm, uimm, jimm, zimm


def test_imm_gen_wire():
    class DUT(m.Circuit):
        io = m.IO(done=m.Out(m.Bit)) + m.ClockIO()
        imm = ImmGenWire(32)()
        ctrl = Control(32)()

        counter = mantle.CounterModM(len(insts), len(insts).bit_length())
        i = m.mux([iimm(i) for i in insts], counter.O)
        s = m.mux([simm(i) for i in insts], counter.O)
        b = m.mux([bimm(i) for i in insts], counter.O)
        u = m.mux([uimm(i) for i in insts], counter.O)
        j = m.mux([jimm(i) for i in insts], counter.O)
        z = m.mux([zimm(i) for i in insts], counter.O)
        x = m.mux([iimm(i) & -2 for i in insts], counter.O)

        out = m.mux([
            m.mux([
                m.mux([
                    m.mux([
                        m.mux([
                            m.mux([
                                x,
                                z
                            ], ctrl.imm_sel == IMM_Z),
                            j
                        ], ctrl.imm_sel == IMM_J),
                        u
                    ], ctrl.imm_sel == IMM_U),
                    b
                ], ctrl.imm_sel == IMM_B),
                s
            ], ctrl.imm_sel == IMM_S),
            i
        ], ctrl.imm_sel == IMM_I)
        inst = m.mux(insts, counter.O)
        ctrl.inst @= inst
        imm.inst @= inst
        imm.sel @= ctrl.imm_sel
        io.done @= counter.COUT

        f.assert_immediate(imm.out == out, failure_msg=(
            "Counter: %d, Type: 0x%x, Out: %x ?= %x", counter.O, imm.sel,
            imm.out, out))
        m.display("Counter: %d, Type: 0x%x, Out: %x ?= %x",
                  counter.O, imm.sel, imm.out, out)

    tester = f.Tester(DUT, DUT.CLK)
    tester.wait_until_high(DUT.done)
    tester.compile_and_run("verilator", magma_opts={"verilator_compat": True,
                                                    "inline": True,
                                                    "terminate_unused": True},
                           flags=['--assert'],
                           disp_type="realtime")
