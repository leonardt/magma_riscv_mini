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
    def __init__(self, data_width: int):
        self.io = m.IO(A=m.In(m.UInt[data_width]), B=m.In(m.UInt[data_width]),
                       op=m.In(ALUOP), out=m.Out(m.UInt[data_width]),
                       sum_=m.Out(m.UInt[data_width]))


class ALUSimple(ALUBase):
    def __init__(self, data_width: int):
        super().__init__(data_width)
        io = self.io

        @m.inline_combinational()
        def alu():
            if io.op == ALUOP.ADD:
                io.out @= io.A + io.B
            elif io.op == ALUOP.SUB:
                io.out @= io.A - io.B
            elif io.op == ALUOP.SRA:
                io.out @= m.uint(m.sint(io.A) >> io.B)
            elif io.op == ALUOP.SRL:
                io.out @= io.A >> io.B
            elif io.op == ALUOP.SLL:
                io.out @= io.A << io.B
            elif io.op == ALUOP.SLT:
                io.out @= m.uint(m.sint(io.A) < m.sint(io.B), 16)
            elif io.op == ALUOP.SLTU:
                io.out @= m.uint(io.A < io.B, 16)
            elif io.op == ALUOP.AND:
                io.out @= io.A & io.B
            elif io.op == ALUOP.OR:
                io.out @= io.A | io.B
            elif io.op == ALUOP.XOR:
                io.out @= io.A ^ io.B
            elif io.op == ALUOP.COPY_A:
                io.out @= io.A
            else:
                io.out @= io.B

            io.sum_ @= io.A + m.mux([io.B, -io.B], io.op[0])


class ALUArea(ALUBase):
    def __init__(self, data_width: int):
        super().__init__(data_width)
        io = self.io
        sum_ = io.A + m.mux([io.B, -io.B], io.op[0])
        cmp = m.uint(m.mux([m.mux([io.A[-1], io.B[-1]], io.op[1]), sum_[-1]],
                           io.A[-1] == io.B[-1]), 16)
        shin = m.mux([io.A[::-1], io.A], io.op[3])
        shiftr = m.uint(m.sint(
            m.concat(shin, io.op[0] & shin[data_width - 1])
        ) >> m.zext(io.B, 1))[:data_width]
        shiftl = shiftr[::-1]

        @m.inline_combinational()
        def alu():
            if (io.op == ALUOP.ADD) | (io.op == ALUOP.SUB):
                io.out @= sum_
            elif (io.op == ALUOP.SLT) | (io.op == ALUOP.SLTU):
                io.out @= cmp
            elif (io.op == ALUOP.SRA) | (io.op == ALUOP.SRL):
                io.out @= shiftr
            elif io.op == ALUOP.SLL:
                io.out @= shiftl
            elif io.op == ALUOP.AND:
                io.out @= io.A & io.B
            elif io.op == ALUOP.OR:
                io.out @= io.A | io.B
            elif io.op == ALUOP.XOR:
                io.out @= io.A ^ io.B
            elif io.op == ALUOP.COPY_A:
                io.out @= io.A
            else:
                io.out @= io.B
        io.sum_ @= sum_
