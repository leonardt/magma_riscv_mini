import operator

import magma as m
from mantle import RegFileBuilder


class RegFile(m.Generator2):
    """
    Basic two read port, one write port register file

    Register at address 0 always holds the value 0
    """
    def __init__(self, DATAWIDTH: int):
        self.io = io = m.IO(
            raddr1=m.In(m.UInt[5]),
            raddr2=m.In(m.UInt[5]),
            rdata1=m.Out(m.UInt[DATAWIDTH]),
            rdata2=m.Out(m.UInt[DATAWIDTH]),
            wen=m.In(m.Enable),
            waddr=m.In(m.UInt[5]),
            wdata=m.In(m.UInt[DATAWIDTH])
        ) + m.ClockIO(has_async_reset=True)
        regs = RegFileBuilder("reg_file", 32, DATAWIDTH, write_forward=False)
        io.rdata1 @= m.mux([0, regs[io.raddr1]], m.reduce(operator.or_,
                                                          io.raddr1))
        io.rdata2 @= m.mux([0, regs[io.raddr2]], m.reduce(operator.or_,
                                                          io.raddr2))
        wen = m.bit(io.wen) & m.reduce(operator.or_, io.waddr)
        regs.write(io.waddr, io.wdata, enable=wen)
