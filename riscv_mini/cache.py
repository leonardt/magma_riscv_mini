import magma as m
from riscv_mini.nasti import make_NastiIO, NastiParameters


def make_CacheReq(x_len):
    class CacheReq(m.Product):
        addr = m.UInt[x_len]
        data = m.UInt[x_len]
        mask = m.UInt[x_len // 8]
    return CacheReq


def make_CacheResp(x_len):
    class CacheResp(m.Product):
        data = m.UInt[x_len]
    return CacheResp


def make_CacheIO(x_len):
    class CacheIO(m.Product):
        abort = m.In(m.Bit)
        req = m.Out(m.Valid[make_CacheReq(x_len)])
        resp = m.In(m.Valid[make_CacheResp(x_len)])
    return CacheIO


def make_CacheModuleIO(x_len):
    class CacheModuleIO(m.Product):
        cpu = make_CacheIO(x_len)
        nasti = make_NastiIO(
            NastiParameters(data_bits=64, addr_bits=x_len, id_bits=5)
        )
    return CacheModuleIO
