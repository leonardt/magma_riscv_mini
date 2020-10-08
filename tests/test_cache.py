from riscv_mini.cache import make_CacheModuleIO
from riscv_mini.nasti import NastiParameters


def test_make_cache_module_io():
    x_len = 32
    make_CacheModuleIO(x_len, NastiParameters(data_bits=64, addr_bits=x_len,
                                              id_bits=5))
