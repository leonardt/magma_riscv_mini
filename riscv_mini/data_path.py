import magma as m


from riscv_mini.alu import ALUArea
from riscv_mini.csr_gen import CSRGen
from riscv_mini.control import make_ControlIO
from riscv_mini.const import Const
from riscv_mini.cache import make_CacheIO
from riscv_mini.imm_gen import ImmGenWire
from riscv_mini.reg_file import RegFile


def make_HostIO(x_len):
    class HostIO(m.Product):
        fromhost = m.In(m.Valid[m.UInt[x_len]])
        tohost = m.Out(m.UInt[x_len])
    return HostIO


def make_DatapathIO(x_len):
    return m.IO(
        host=make_HostIO(x_len),
        icache=m.Flip(make_CacheIO(x_len)),
        dcache=m.Flip(make_CacheIO(x_len)),
        ctrl=m.Flip(make_ControlIO(x_len))
    )


class Datapath(m.Generator2):
    def __init__(self, x_len, ALU=ALUArea, ImmGen=ImmGenWire):
        self.io = make_DatapathIO(x_len)
        csr = CSRGen(x_len)
        reg_file = RegFile(x_len)
        alu = ALU(x_len)
        imm_gen = ImmGen(x_len)
