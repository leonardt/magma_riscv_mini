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


def make_CacheModuleIO(x_len, nasti_params):
    class CacheModuleIO(m.Product):
        cpu = make_CacheIO(x_len)
        nasti = make_NastiIO(nasti_params)
    return CacheModuleIO


class Cache(m.Generator2):
    def __init__(self, x_len, n_ways: int, n_sets: int, b_bytes: int):
        b_bits = b_bytes << 3
        b_len = m.bitutils.clog2(b_bytes)
        s_len = m.bitutils.clog2(n_sets)
        t_len = x_len - (s_len + b_len)
        n_words = b_bits // x_len
        w_bytes = x_len // 8
        byte_offset_bits = m.bitutils.clog2(w_bytes)
        nasti_params = NastiParameters(data_bits=64, addr_bits=x_len,
                                       id_bits=5)
        data_baets = b_bits // nasti_params.x_data_bits
