import magma as m

from riscv_mini.cache import make_CacheIO
from riscv_mini.data_path import Datapath, make_HostIO
from riscv_mini.control import Control


class Core(m.Generator2):
    def __init__(self, x_len):
        self.io = m.IO(
            host=make_HostIO(x_len),
            icache=m.Flip(make_CacheIO(x_len)),
            dcache=m.Flip(make_CacheIO(x_len)),
        ) + m.ClockIO(has_reset=True)

        data_path = Datapath(x_len)()
        control = Control(x_len)()

        m.wire(self.io.host, data_path.host)
        m.wire(data_path.icache, self.io.icache)
        m.wire(data_path.dcache, self.io.dcache)
        for name, value in data_path.ctrl.items():
            m.wire(value, getattr(control, name))
