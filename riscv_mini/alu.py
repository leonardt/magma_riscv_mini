import magma as m


class ALUOP(m.Enum):
    ADD = 0
    SUB = 1
    AND = 2
    OR = 3
    XOR = 4
    SLT = 5
    SLL = 6
    SLTU = 7
    SRL = 8
    SRA = 9
    COPY_A = 10
    COPY_B = 11
    XXX = 15


class ALUBase(m.Generator2):
    def __init__(self, x_len: int):
        self.io = m.IO(A=m.In(m.UInt[x_len]), B=m.In(m.UInt[x_len]),
                       op=m.In(ALUOP), O=m.Out(m.UInt[x_len]),
                       sum_=m.Out(m.UInt[x_len]))


class ALUSimple(ALUBase):
    def __init__(self, x_len: int):
        super().__init__(x_len)
        io = self.io

        @m.inline_combinational()
        def alu():
            if io.op == ALUOP.ADD:
                io.O @= io.A + io.B
            elif io.op == ALUOP.SUB:
                io.O @= io.A - io.B
            elif io.op == ALUOP.SRA:
                io.O @= m.uint(m.sint(io.A) >> m.sint(io.B))
            elif io.op == ALUOP.SRL:
                io.O @= io.A >> io.B
            elif io.op == ALUOP.SLL:
                io.O @= io.A << io.B
            elif io.op == ALUOP.SLT:
                io.O @= m.uint(m.sint(io.A) < m.sint(io.B), x_len)
            elif io.op == ALUOP.SLTU:
                io.O @= m.uint(io.A < io.B, x_len)
            elif io.op == ALUOP.AND:
                io.O @= io.A & io.B
            elif io.op == ALUOP.OR:
                io.O @= io.A | io.B
            elif io.op == ALUOP.XOR:
                io.O @= io.A ^ io.B
            elif io.op == ALUOP.COPY_A:
                io.O @= io.A
            else:
                io.O @= io.B

            io.sum_ @= io.A + m.mux([io.B, -io.B], io.op[0])


class ALUArea(ALUBase):
    def __init__(self, x_len: int):
        super().__init__(x_len)
        io = self.io
        sum_ = io.A + m.mux([io.B, -io.B], io.op[0])
        cmp = m.uint(m.mux([m.mux([io.A[-1], io.B[-1]], io.op[1]), sum_[-1]],
                           io.A[-1] == io.B[-1]), x_len)
        shin = m.mux([io.A[::-1], io.A], io.op[3])
        x = m.sint(
            m.concat2(shin, m.Bits[1](io.op[0] & shin[x_len - 1]))
        ) >> m.sint(m.zext(io.B, 1))
        shiftr = m.uint(m.sint(
            m.concat2(shin, m.Bits[1](io.op[0] & shin[x_len - 1]))
        ) >> m.sint(m.zext(io.B, 1)))[:x_len]
        shiftl = shiftr[::-1]

        @m.inline_combinational()
        def alu():
            if (io.op == ALUOP.ADD) | (io.op == ALUOP.SUB):
                io.O @= sum_
            elif (io.op == ALUOP.SLT) | (io.op == ALUOP.SLTU):
                io.O @= cmp
            elif (io.op == ALUOP.SRA) | (io.op == ALUOP.SRL):
                io.O @= shiftr
            elif io.op == ALUOP.SLL:
                io.O @= shiftl
            elif io.op == ALUOP.AND:
                io.O @= io.A & io.B
            elif io.op == ALUOP.OR:
                io.O @= io.A | io.B
            elif io.op == ALUOP.XOR:
                io.O @= io.A ^ io.B
            elif io.op == ALUOP.COPY_A:
                io.O @= io.A
            else:
                io.O @= io.B
        io.sum_ @= sum_
