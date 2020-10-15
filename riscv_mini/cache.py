import magma as m
import mantle
from riscv_mini.nasti import (make_NastiIO, NastiParameters,
                              NastiReadAddressChannel,
                              NastiWriteAddressChannel, NastiWriteDataChannel)
m.config.set_debug_mode(True)


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


class ArrayMaskMem(m.Generator2):
    """
    Wrapper around a memory to store entries containing an array of values that
    can be written using a write mask for each array index

    Implemented using a separate memory for each array index, the mask indices
    are mapped into the WEN ports of each sub-memory
    """

    def __init__(self, height, array_length, T, read_latency, has_read_enable):
        addr_width = m.bitutils.clog2(height)
        self.io = m.IO(
            RADDR=m.In(m.Bits[addr_width]),
            RDATA=m.Out(m.Array[array_length, T]),
        ) + m.ClockIO()
        if has_read_enable:
            self.io += m.IO(RE=m.In(m.Enable))
        self.io += m.IO(
            WADDR=m.In(m.Bits[addr_width]),
            WDATA=m.In(m.Array[array_length, T]),
            WMASK=m.In(m.Bits[array_length]),
            WE=m.In(m.Enable)
        )
        for i in range(array_length):
            mem = m.Memory(height, T, read_latency,
                           has_read_enable=has_read_enable)()
            mem.RADDR @= self.io.RADDR
            if has_read_enable:
                mem.RE @= self.io.RE
            self.io.RDATA[i] @= mem.RDATA

            mem.write(self.io.WDATA[i], self.io.WADDR,
                      m.enable(m.bit(self.io.WE) & self.io.WMASK[i]))

        def read(self, addr, enable=None):
            self.RADDR @= addr
            if enable is not None:
                if not has_read_enable:
                    raise Exception("Cannot use `enable` with no read enable")
                self.RE @= enable
            return self.RDATA

        self.read = read

        def write(self, data, addr, mask, enable):
            self.WDATA @= data
            self.WADDR @= addr
            self.WMASK @= mask
            self.WE @= enable

        self.write = write


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
        v = m.Register(m.UInt[n_sets], has_enable=True)()
        d = m.Register(m.UInt[n_sets], has_enable=True)()
        meta_mem = m.Memory(n_sets, MetaData, read_latency=1,
                            has_read_enable=True)()
        data_mem = [ArrayMaskMem(n_sets, w_bytes, m.UInt[8], read_latency=1,
                                 has_read_enable=True)()
                    for _ in range(n_words)]

        addr_reg = m.Register(type(self.io.cpu.req.data.addr).as_undirected(),
                              has_enable=True)()
        cpu_data = m.Register(type(self.io.cpu.req.data.data).as_undirected(),
                              has_enable=True)()
        cpu_mask = m.Register(type(self.io.cpu.req.data.mask).as_undirected(),
                              has_enable=True)()

        self.io.nasti.w.valid.undriven()

        self.io.nasti.r.ready @= state.O == State.REFILL
        # Counters
        assert data_beats > 0
        if data_beats > 1:
            counter_m = data_beats - 1
            read_counter = mantle.CounterModM(counter_m,
                                              max(counter_m.bit_length(), 1),
                                              has_ce=True)
            read_counter.CE @= m.enable(self.io.nasti.r.fired())
            read_count, read_wrap_out = read_counter.O, read_counter.COUT
        else:
            read_count, read_wrap_out = 0, 1

        refill_buf = m.Register(
            m.Array[data_beats, m.UInt[nasti_params.x_data_bits]],
            has_enable=True
        )()
        if data_beats == 1:
            refill_buf.I[0] @= self.io.nasti.r.data.data
        else:
            refill_buf.I @= m.set_index(refill_buf.O,
                                        self.io.nasti.r.data.data,
                                        read_count)
        refill_buf.CE @= m.enable(self.io.nasti.r.fired())

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

        read = m.mux([
            m.as_bits(m.mux([
                rdata_buf,
                rdata
            ], ren_reg)),
            m.as_bits(refill_buf.O)
        ], is_alloc_reg)

        hit @= v.O[idx_reg] & (rmeta.tag == tag_reg)

        # read mux
        self.io.cpu.resp.data.data @= m.array(
            [read[i * x_len:(i + 1) * x_len] for i in range(n_words)]
        )[off_reg]
        self.io.cpu.resp.valid @= (is_idle | (is_read & hit) |
                                   (is_alloc_reg & ~cpu_mask.O.reduce_or()))

        addr_reg.I @= addr
        addr_reg.CE @= m.enable(self.io.cpu.resp.valid.value())

        cpu_data.I @= self.io.cpu.req.data.data
        cpu_data.CE @= m.enable(self.io.cpu.resp.valid.value())

        cpu_mask.I @= self.io.cpu.req.data.mask
        cpu_mask.CE @= m.enable(self.io.cpu.resp.valid.value())

        wmeta = MetaData(name="wmeta")
        wmeta.tag @= tag_reg

        offset_mask = cpu_mask.O << m.concat(off_reg,
                                             m.bits(0, byte_offset_bits))
        wmask = m.mux([
            m.zext_to(offset_mask, w_bytes * 8),
            m.SInt[w_bytes * 8](-1)
        ], ~is_alloc)

        if len(refill_buf.O) == 1:
            wdata_alloc = self.io.nasti.r.data.data
        else:
            wdata_alloc = m.concat(
                self.io.nasti.r.data.data,
                # TODO: not sure why they use `init.reverse`
                # https://github.com/ucb-bar/riscv-mini/blob/release/src/main/scala/Cache.scala#L116
                # TODO: Needed to drop first index here to match type with
                # other mux input?
                m.concat(*reversed(refill_buf.O[1:]))
            )
        wdata = m.mux([
            wdata_alloc,
            m.as_bits(m.repeat(cpu_data.O, n_words))
        ], ~is_alloc)

        v.I @= m.set_index(v.O, m.bit(True), idx_reg)
        v.CE @= m.enable(wen)
        d.I @= m.set_index(d.O, ~is_alloc, idx_reg)
        d.CE @= m.enable(wen)

        meta_mem.write(wmeta, idx_reg, m.enable(wen & is_alloc))
        for i, mem in enumerate(data_mem):
            data = [wdata[i * x_len + j * 8:i * x_len + (j + 1) * 8]
                    for j in range(w_bytes)]
            mem.write(m.array(data), idx_reg,
                      wmask[i * w_bytes: (i + 1) * w_bytes], m.enable(wen))

        tag_and_idx = m.zext_to(m.concat(tag_reg, idx_reg),
                                nasti_params.x_addr_bits)
        self.io.nasti.ar.data @= NastiReadAddressChannel(
            nasti_params, 0, tag_and_idx << m.Bits[len(tag_and_idx)](b_len),
            m.bitutils.clog2(nasti_params.x_data_bits // 8), data_beats - 1)
        # TODO: Default ar.valid
        # io.nasti.ar.valid @= False

        rmeta_and_idx = m.zext_to(m.concat(rmeta.tag, idx_reg),
                                  nasti_params.x_addr_bits)
        self.io.nasti.aw.data @= NastiWriteAddressChannel(
            nasti_params, 0, rmeta_and_idx <<
            m.Bits[len(rmeta_and_idx)](b_len),
            m.bitutils.clog2(nasti_params.x_data_bits // 8), data_beats - 1)
        # io.nasti.aw.valid @= False
