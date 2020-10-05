import random
from hwtypes import BitVector, Bit
from .opcode import Opcode, Funct3


def reg(x):
    return BitVector[5](x & ((1 << 5) - 1))


def imm(x):
    if isinstance(x, int):
        x = BitVector[21](x)
    else:
        assert isinstance(x, BitVector)
        x = x[:21]
    return (x & ((1 << 20) - 1))


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


def I(funct3, rd, rs1, i):
    return concat(imm(i)[0:12], reg(rs1), funct3, reg(rd), Opcode.ITYPE)


def SYS(funct3, rd, csr, rs1):
    return concat(csr, reg(rs1), funct3, reg(rd), Opcode.SYSTEM)


def J(rd, i):
    return concat(imm(i)[20], imm(i)[1:11], imm(i)[11], imm(i)[12:20], reg(rd),
                  Opcode.JAL)


def L(funct3, rd, rs1, i):
    return concat(imm(i)[0:12], reg(rs1), funct3, reg(rd), Opcode.LOAD)


def JR(rd, rs1, i):
    return concat(imm(i)[0:12], reg(rs1), BitVector[3](0), reg(rd),
                  Opcode.JALR)


rand_fn3 = BitVector.random(3)
rand_rs1 = BitVector.random(5) + 1
rand_rs2 = BitVector.random(5) + 1
rand_rd = BitVector.random(5) + 1
rand_inst = BitVector.random(32)


def csr(inst):
    return (inst >> 20)[:12]


def rs1(inst):
    return ((inst >> 15) & 0x1f)[:12]


nop = concat(BitVector[12](0), reg(0), Funct3.ADD, reg(0), Opcode.ITYPE)
