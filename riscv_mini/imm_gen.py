import magma as m
from riscv_mini.control import IMM_I, IMM_S, IMM_B, IMM_U, IMM_J, IMM_Z


class ImmGen(m.Generator2):
    def __init__(self, x_len):
        self.io = m.IO(
            inst=m.In(m.UInt[x_len]),
            sel=m.In(m.UInt[3]),
            O=m.Out(m.UInt[x_len])
        )


class ImmGenWire(ImmGen):
    def __init__(self, x_len):
        super().__init__(x_len)
        inst = self.io.inst
        Iimm = m.sext_to(m.sint(inst[20:32]), x_len)
        Simm = m.sext_to(m.sint(m.concat(inst[7:12], inst[25:32])), x_len)
        Bimm = m.sext_to(m.sint(m.concat(
            m.bits(0, 1), inst[8:12], inst[25:31], inst[7], inst[31]
        )), x_len)
        Uimm = m.concat(m.bits(0, 12), inst[12:32])
        Jimm = m.sext_to(m.sint(m.concat(
            m.bits(0, 1), inst[21:25], inst[25:31], inst[20], inst[12:20],
            inst[31]
        )), x_len)
        Zimm = m.zext_to(inst[15:20], x_len)

        self.io.O @= m.dict_lookup({
            IMM_I: Iimm,
            IMM_S: Simm,
            IMM_B: Bimm,
            IMM_U: Uimm,
            IMM_J: Jimm,
            IMM_Z: Zimm
        }, self.io.sel, Iimm & -2)


# TODO: ImmGenMux
