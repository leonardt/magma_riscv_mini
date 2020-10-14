import magma as m
import mantle
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
        req = m.In(m.Valid[make_CacheReq(x_len)])
        resp = m.Out(m.Valid[make_CacheResp(x_len)])
    return CacheIO


def make_cache_ports(x_len, nasti_params):
    return {
        "cpu": make_CacheIO(x_len),
        "nasti": make_NastiIO(nasti_params)
    }


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
        data_beats = b_bits // nasti_params.x_data_bits

        class MetaData(m.Product):
            tag = m.UInt[t_len]

        self.io = m.IO(**make_cache_ports(x_len, nasti_params))
        self.io += m.ClockIO()

        class State(m.Enum):
            IDLE = 0
            READ_CACHE = 1
            WRITE_CACHE = 2
            WRITE_BACK = 3
            WRITE_ACK = 4
            REFILL_READY = 5
            REFILL = 6

        state = m.Register(init=State.IDLE)()

        # memory
        v = m.Register(m.UInt[n_sets])
        d = m.Register(m.UInt[n_sets])
        meta_mem = m.Memory(n_sets, MetaData, read_latency=1,
                            has_read_enable=True)()
        data_mem = [m.Memory(n_sets, m.Array[w_bytes, m.UInt[8]],
                             read_latency=1, has_read_enable=True)()
                    for _ in range(n_words)]

        addr_reg = m.Register(type(self.io.cpu.req.data.addr).as_undirected())
        cpu_data = m.Register(type(self.io.cpu.req.data.data).as_undirected())
        cpu_mask = m.Register(type(self.io.cpu.req.data.mask).as_undirected())

        # TODO: Temporary stub
        self.io.nasti.r.ready.undriven()
        self.io.nasti.w.valid.undriven()

        # Counters
        assert data_beats > 0
        read_counter = mantle.CounterModM(data_beats,
                                          max(data_beats.bit_length(), 1),
                                          has_ce=True)
        read_counter.CE @= m.enable(self.io.nasti.r.fired())
        read_count, read_wrap_out = read_counter.O, read_counter.COUT
        write_counter = mantle.CounterModM(data_beats,
                                           max(data_beats.bit_length(), 1),
                                           has_ce=True)
        write_counter.CE @= m.enable(self.io.nasti.w.fired())
        write_count, write_wrap_out = write_counter.O, write_counter.COUT

        is_idle = state.O == State.IDLE
        is_read = state.O == State.READ_CACHE
        is_write = state.O == State.WRITE_CACHE
        is_alloc = (state.O == State.REFILL) & read_wrap_out
        is_alloc_reg = m.Register(m.Bit)()(is_alloc)

        hit = m.Bit(name="hit")
        wen = is_write & (hit | is_alloc_reg) & ~self.io.cpu.abort | is_alloc
        ren = m.enable(~wen & (is_idle | is_read) & self.io.cpu.req.valid)
        ren_reg = m.enable(m.Register(m.Bit)()(ren))

        addr = self.io.cpu.req.data.addr
        idx = addr[b_len:s_len + b_len]
        tag_reg = addr_reg.O[s_len + b_len:x_len]
        idx_reg = addr_reg.O[b_len:s_len + b_len]
        off_reg = addr_reg.O[byte_offset_bits:b_len]

        rmeta = meta_mem.read(idx, ren)
        rdata = m.concat(*reversed(
            tuple(mem.read(idx, ren) for mem in data_mem)
        ))
        rdata_buf = m.Register(type(rdata), has_enable=True)()(rdata,
                                                               CE=ren_reg)
        refill_buf = m.Register(
            m.Array[data_beats, m.UInt[nasti_params.x_data_bits]]
        )()

        read = m.mux([
            m.as_bits(m.mux([
                rdata_buf,
                rdata
            ], ren_reg)),
            m.as_bits(refill_buf.O)
        ], is_alloc_reg)
