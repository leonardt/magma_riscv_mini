import pytest

import magma as m
from mantle import CounterModM
import fault
from hwtypes import BitVector as BV

from riscv_mini.control import (Control, BR_EQ, BR_NE, BR_LT, BR_GE, BR_LTU,
                                BR_GEU)
from riscv_mini.br_cond import BrCondArea, BrCondSimple

from .utils import B
from .opcode import Funct3


@pytest.mark.parametrize("BrCond", [BrCondSimple, BrCondArea])
def test_br_cond(BrCond):
    x_len = 32

    class BrCond_DUT(m.Circuit):
        _IGNORE_UNUSED_ = True
        io = m.IO(
            done=m.Out(m.Bit),
            out=m.Out(m.Bit),
            taken=m.Out(m.Bit)
        ) + m.ClockIO()
        br_cond = BrCond(x_len)()
        io.taken @= br_cond.taken

        control = Control(x_len)()
        br_cond.br_type @= control.br_type

        insts = [
            B(Funct3.BEQ, 0, 0, 0),
            B(Funct3.BNE, 0, 0, 0),
            B(Funct3.BLT, 0, 0, 0),
            B(Funct3.BGE, 0, 0, 0),
            B(Funct3.BLTU, 0, 0, 0),
            B(Funct3.BGEU, 0, 0, 0),
        ] * 10

        n = len(insts)
        counter = CounterModM(n, n.bit_length())
        control.inst @= m.mux(insts, counter.O)
        io.done @= counter.COUT

        rs1 = [BV.random(x_len) for _ in range(n)]
        rs2 = [BV.random(x_len) for _ in range(n)]
        br_cond.rs1 @= m.mux(rs1, counter.O)
        br_cond.rs2 @= m.mux(rs2, counter.O)

        eq = [a == b for a, b in zip(rs1, rs2)]
        ne = [a != b for a, b in zip(rs1, rs2)]
        lt = [m.sint(a) < m.sint(b) for a, b in zip(rs1, rs2)]
        ge = [m.sint(a) >= m.sint(b) for a, b in zip(rs1, rs2)]
        ltu = [a < b for a, b in zip(rs1, rs2)]
        geu = [a >= b for a, b in zip(rs1, rs2)]

        @m.inline_combinational()
        def logic():
            if control.br_type == BR_EQ:
                io.out @= m.mux(eq, counter.O)
            elif control.br_type == BR_NE:
                io.out @= m.mux(ne, counter.O)
            elif control.br_type == BR_LT:
                io.out @= m.mux(lt, counter.O)
            elif control.br_type == BR_GE:
                io.out @= m.mux(ge, counter.O)
            elif control.br_type == BR_LTU:
                io.out @= m.mux(ltu, counter.O)
            elif control.br_type == BR_GEU:
                io.out @= m.mux(geu, counter.O)
            else:
                io.out @= False

    tester = fault.Tester(BrCond_DUT, BrCond_DUT.CLK)
    tester.wait_until_high(tester.circuit.done)
    tester.assert_(tester.circuit.taken == tester.circuit.out)
    tester.compile_and_run("verilator", magma_opts={"verilator_compat": True,
                                                    "inline": True,
                                                    "terminate_unused": True})
