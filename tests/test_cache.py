import magma as m

from riscv_mini.nasti import make_NastiIO, NastiParameters
from riscv_mini.cache import Cache, make_CacheResp, make_CacheReq


class GoldCache(m.Generator2):
    def __init__(self, x_len, n_ways: int, n_sets: int, b_bytes: int):
        nasti_params = NastiParameters(data_bits=64, addr_bits=x_len,
                                       id_bits=5)

        self.io = m.IO(
            req=m.Consumer(m.Decoupled[make_CacheReq(x_len)]),
            resp=m.Producer(m.Decoupled[make_CacheResp(x_len)]),
            nasti=make_NastiIO(nasti_params)
        )
        size = m.bitutils.clog2(nasti_params.x_data_bits)
        b_bits = b_bytes << 3
        b_len = m.bitutils.clog2(b_bytes)
        s_len = m.bitutils.clog2(n_sets)
        t_len = x_len - (s_len + b_len)
        n_words = b_bits // x_len
        w_bytes = x_len // 8
        byte_offset_bits = m.bitutils.clog2(w_bytes)
        nasti_params = NastiParameters(data_bits=64, addr_bits=x_len,
                                       id_bits=5)
        data_beats = b_bits // nasti_params.x_data_bits
        length = data_beats - 1

        data = m.Memory(n_sets, m.UInt[b_bits])()
        tags = m.Memory(n_sets, m.UInt[t_len])()
        v = m.Memory(n_sets, m.Bit)()
        d = m.Memory(n_sets, m.Bit)()

        req = self.io.req.data
        tag = req.addr >> (b_len + s_len)
        idx = req.addr[b_len:b_len + s_len]
        off = req.addr[:b_len]
        read = data.read(idx)
        write = m.bits(0, b_bits)
        for i in range(b_bytes):
            write |= m.mux([
                (read & (0xff << (8 * i))),
                ((m.zext_to(req.data, b_bits) >> ((8 * (i & 0x3)))) &
                 0xff) << (8 * i)
            ], ((off // 4) == (i // 4)) & (req.mask >> (i & 0x3))[0])[:b_bits]


        self.io.resp.data.data @= (read >> m.zext_to((off // 4) * x_len,
                                                     b_bits))[:x_len]


def test_cache():
    class DUT(m.Circuit):
        io = m.IO(O=m.Out(m.Bit))
        MyCache = Cache(32, 1, 256, 4 * (32 >> 3))()
        GoldCache(32, 1, 256, 4 * (32 >> 3))()

    m.compile("build/CacheDUT", DUT, inline=True, drive_undriven=True,
              terminate_unused=True)
