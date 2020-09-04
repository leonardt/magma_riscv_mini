from hwtypes import BitVector, Bit
from .opcode import Opcode


def reg(x):
    return BitVector[5](x & ((1 << 5) - 1))


def imm(x):
    return BitVector[21](x & ((1 << 20) - 1))


def concat(*args):
    x = args[0]
    if isinstance(x, Bit):
        x = BitVector[1](x)
    if len(args) == 1:
        return x
    return x.concat(concat(*args[1:]))


def B(funct3, rs1, rs2, i):
    return concat(imm(i)[12], imm(i)[5:11], reg(rs2), reg(rs1), funct3,
                  imm(i)[1:5], imm(i)[11], Opcode.BRANCH)
