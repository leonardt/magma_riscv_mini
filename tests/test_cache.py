import magma as m

from riscv_mini.cache import Cache


def test_cache():
    MyCache = Cache(32, 1, 256, 4 * (32 >> 3))
    m.compile("build/MyCache", MyCache, inline=True,
              drive_undriven=True, terminate_unused=True)
