import magma as m

from riscv_mini.cache import Cache
from riscv_mini.nasti import NastiParameters


def test_cache():
    MyCache = Cache(32, 4, 4, 4)
    m.compile("build/MyCache", MyCache, inline=True,
              drive_undriven=True, terminate_unused=True)
