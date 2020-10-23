import magma as m
import mantle

from riscv_mini.nasti import (make_NastiIO, NastiParameters,
                              NastiReadAddressChannel,
                              NastiWriteAddressChannel, NastiWriteDataChannel)
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
        nasti_params = NastiParameters(data_bits=64, addr_bits=x_len,
                                       id_bits=5)
        data_beats = b_bits // nasti_params.x_data_bits
        length = data_beats - 1

        data = m.Memory(n_sets, m.UInt[b_bits])()
        tags = m.Memory(n_sets, m.UInt[t_len])()
        v = m.Memory(n_sets, m.Bit)()
        d = m.Memory(n_sets, m.Bit)()

        req = self.io.req.data
        tag = (req.addr >> (b_len + s_len))[:t_len]
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

        class State(m.Enum):
            IDLE = 0
            WRITE = 1
            WRITE_ACK = 1
            READ = 1

        state = m.Register(init=State.IDLE)()

        counter_m = data_beats - 1
        read_counter = mantle.CounterModM(counter_m,
                                          max(counter_m.bit_length(), 1),
                                          has_ce=True)
        read_counter.CE @= m.enable(state.O == State.READ)
        r_cnt, r_done = read_counter.O, read_counter.COUT

        write_counter = mantle.CounterModM(counter_m,
                                           max(counter_m.bit_length(), 1),
                                           has_ce=True)
        write_counter.CE @= m.enable(state.O == State.WRITE &
                                     self.io.nasti.r.valid)
        w_cnt, w_done = write_counter.O, write_counter.COUT

        self.io.resp.data.data @= (read >> m.zext_to((off // 4) * x_len,
                                                     b_bits))[:x_len]
        self.io.nasti.ar.data @= NastiReadAddressChannel(
            nasti_params, 0, (req.addr >> b_len) << b_len, size, length)
        tags_rdata = tags.read(idx)
        self.io.nasti.aw.data @= NastiWriteAddressChannel(
            nasti_params, 0,
            m.bits(m.concat(tags_rdata, idx), 32) << b_len, size, length)
        self.io.nasti.w.data @= NastiWriteDataChannel(
            nasti_params,
            (read >> (m.zext_to(w_cnt, b_bits) *
                      nasti_params.x_data_bits))[:nasti_params.x_data_bits],
            None, w_done)
        self.io.nasti.w.valid @= state.O == State.WRITE
        self.io.nasti.b.ready @= state.O == State.WRITE_ACK
        self.io.nasti.r.ready @= state.O == State.READ

        d_wen = m.Bit(name="d_wen")
        d.write(True, idx, m.enable(d_wen))

        data_wen = m.Bit(name="data_wen")
        data_wdata = m.UInt[b_bits](name="data_wdata")
        data.write(data_wdata, idx, m.enable(data_wen))

        v_wen = m.Bit(name="v_wen")
        v.write(True, idx, m.enable(v_wen))
        v_rdata = v.read(idx)

        tags_wen = m.Bit(name="tags_wen")
        tags.write(tag, idx, m.enable(tags_wen))

        d_rdata = d.read(idx)

        @m.inline_combinational()
        def logic():
            self.io.resp.valid @= False
            self.io.req.ready @= False
            self.io.nasti.ar.valid @= False
            self.io.nasti.aw.valid @= False

            d_wen @= False

            data_wen @= False
            data_wdata @= m.UInt[b_bits](0)
            state.I @= state.O

            tags_wen @= False
            v_wen @= False

            if state.O == State.IDLE:
                if self.io.req.valid & self.io.resp.ready:
                    if v_rdata & (tags_rdata == tag):
                        if req.mask.reduce_or():
                            d_wen @= True
                            data_wdata @= write
                        self.io.req.ready @= True
                        self.io.resp.valid @= True
                    else:
                        if d_rdata:
                            self.io.nasti.aw.valid @= True
                            state.I @= State.WRITE
                        else:
                            self.io.nasti.ar.valid @= True
                            state.I @= State.READ
            elif state.O == State.WRITE:
                if w_done:
                    state.I @= State.WRITE_ACK
            elif state.O == State.WRITE_ACK:
                if self.io.nasti.b.valid:
                    data_wdata @= 0
                    data_wen @= True
                    state.I @= State.READ
            elif state.O == State.READ:
                if self.io.nasti.r.valid:
                    data_wdata @= read | (m.zext_to(self.io.nasti.r.data.data,
                                                    b_bits) <<
                                          (m.zext_to(r_cnt, b_bits) *
                                           nasti_params.x_data_bits))
                    data_wen @= True
                if r_done:
                    tags_wen @= True
                    v_wen @= True
                    state.I @= State.IDLE


def test_cache():
    class DUT(m.Circuit):
        io = m.IO(O=m.Out(m.Bit))
        MyCache = Cache(32, 1, 256, 4 * (32 >> 3))()
        GoldCache(32, 1, 256, 4 * (32 >> 3))()

    m.compile("build/CacheDUT", DUT, inline=True, drive_undriven=True,
              terminate_unused=True)
