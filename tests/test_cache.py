import itertools
import random
from hwtypes import BitVector
import magma as m
m.config.set_debug_mode(True)
import mantle
import fault as f

from riscv_mini.nasti import (make_NastiIO, NastiParameters,
                              NastiReadAddressChannel,
                              NastiWriteAddressChannel, NastiWriteDataChannel,
                              NastiWriteResponseChannel, NastiReadDataChannel)
from riscv_mini.cache import Cache, make_CacheResp, make_CacheReq


class Queue(m.Generator2):
    def __init__(self, T, entries, pipe=False, flow=False):
        assert entries >= 0
        self.io = m.IO(
            # Flipped since enq/deq is from perspective of the client
            enq=m.DeqIO[T],
            deq=m.EnqIO[T],
            count=m.Out(m.UInt[m.bitutils.clog2(entries + 1)])
        ) + m.ClockIO()

        ram = m.Memory(entries, T)()
        counter_m = entries - 1
        enq_ptr = mantle.CounterModM(counter_m, counter_m.bit_length(),
                                     has_ce=True)
        deq_ptr = mantle.CounterModM(counter_m, counter_m.bit_length(),
                                     has_ce=True)
        maybe_full = m.Register(init=False, has_enable=True)()

        ptr_match = enq_ptr.O == deq_ptr.O
        empty = ptr_match & ~maybe_full.O
        full = ptr_match & maybe_full.O

        self.io.deq.valid @= ~empty
        self.io.enq.ready @= ~full

        do_enq = m.enable(self.io.enq.fired())
        do_deq = m.enable(self.io.deq.fired())

        ram.write(self.io.enq.data, enq_ptr.O, do_enq)
        enq_ptr.CE @= do_enq

        deq_ptr.CE @= do_deq

        maybe_full.I @= do_enq
        maybe_full.CE @= m.enable(do_enq != do_deq)
        self.io.deq.data @= ram[deq_ptr.O]

        if flow:
            raise NotImplementedError()
        if pipe:
            raise NotImplementedError()

        def ispow2(n):
            return (n & (n - 1) == 0) and n != 0

        ptr_diff = enq_ptr.O - deq_ptr.O
        count_len = len(self.io.count)
        if ispow2(entries):
            self.io.count @= m.mux([m.bits(0, count_len), entries],
                                   maybe_full.O & ptr_match)
        else:
            self.io.count @= m.mux([
                m.mux([
                    m.bits(0, count_len),
                    entries
                ], maybe_full.O),
                m.mux([
                    ptr_diff,
                    entries + ptr_diff
                ], deq_ptr.O > enq_ptr.O)
            ], ptr_match)


def make_Queue(value, entries):
    queue = Queue(type(value.data).as_undirected(), entries)()
    queue.enq.valid @= value.valid
    queue.enq.data @= value.data
    value.ready @= queue.enq.ready
    return queue.deq


class GoldCache(m.Generator2):
    def __init__(self, x_len, n_ways: int, n_sets: int, b_bytes: int):
        nasti_params = NastiParameters(data_bits=64, addr_bits=x_len,
                                       id_bits=5)

        self.io = m.IO(
            req=m.Consumer(m.Decoupled[make_CacheReq(x_len)]),
            resp=m.Producer(m.Decoupled[make_CacheResp(x_len)]),
            nasti=make_NastiIO(nasti_params)
        ) + m.ClockIO()
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
        io = m.IO(done=m.Out(m.Bit)) + m.ClockIO()
        x_len = 32
        n_sets = 256
        b_bytes = 4 * (x_len >> 3)
        b_len = m.bitutils.clog2(b_bytes)
        s_len = m.bitutils.clog2(n_sets)
        t_len = x_len - (s_len + b_len)
        nasti_params = NastiParameters(data_bits=64, addr_bits=x_len,
                                       id_bits=5)

        dut = Cache(x_len, 1, n_sets, b_bytes)()
        dut_mem = make_NastiIO(nasti_params).as_undirected()(name="dut_mem")
        dut_mem.ar @= make_Queue(dut.nasti.ar, 32)
        dut_mem.aw @= make_Queue(dut.nasti.aw, 32)
        dut_mem.w @= make_Queue(dut.nasti.w, 32)
        dut.nasti.b @= make_Queue(dut_mem.b, 32)
        dut.nasti.r @= make_Queue(dut_mem.r, 32)

        gold = GoldCache(x_len, 1, n_sets, b_bytes)()
        gold_req = type(gold.req).as_undirected()(name="gold_req")
        gold_resp = type(gold.resp).as_undirected()(name="gold_resp")
        gold_mem = make_NastiIO(nasti_params).as_undirected()(name="gold_mem")
        gold.req @= make_Queue(gold_req, 32)
        gold_resp @= make_Queue(gold.resp, 32)
        gold_mem.ar @= make_Queue(gold.nasti.ar, 32)
        gold_mem.aw @= make_Queue(gold.nasti.aw, 32)
        gold_mem.w @= make_Queue(gold.nasti.w, 32)
        gold.nasti.b @= make_Queue(gold_mem.b, 32)
        gold.nasti.r @= make_Queue(gold_mem.r, 32)

        size = m.bitutils.clog2(nasti_params.x_data_bits // 8)
        b_bits = b_bytes << 3
        data_beats = b_bits // nasti_params.x_data_bits
        length = data_beats - 1

        mem = m.Memory(1 << 20, m.UInt[nasti_params.x_data_bits])()

        class MemState(m.Enum):
            IDLE = 0
            WRITE = 1
            WRITE_ACK = 2
            READ = 3

        mem_state = m.Register(init=MemState.IDLE)()

        write_counter = mantle.CounterModM(length, length.bit_length(),
                                           has_ce=True)
        write_counter.CE @= m.enable((mem_state.O == MemState.WRITE) &
                                     dut_mem.w.valid & gold_mem.w.valid)
        read_counter = mantle.CounterModM(length, length.bit_length(),
                                          has_ce=True)
        read_counter.CE @= m.enable((mem_state.O == MemState.READ) &
                                    dut_mem.r.ready & gold_mem.r.ready)

        dut_mem.b.valid @= mem_state.O == MemState.WRITE_ACK
        dut_mem.b.data @= NastiWriteResponseChannel(nasti_params, 0)
        dut_mem.r.valid @= mem_state.O == MemState.READ
        dut_mem.r.data @= NastiReadDataChannel(
            nasti_params, 0,
            mem.read(
                ((gold_mem.ar.data.addr >> size) +
                 m.zext_to(read_counter.O, nasti_params.x_addr_bits))[:20]),
            read_counter.COUT)
        gold_mem.ar.ready @= dut_mem.ar.ready
        gold_mem.aw.ready @= dut_mem.aw.ready
        gold_mem.w.ready @= dut_mem.w.ready
        gold_mem.b.valid @= dut_mem.b.valid
        gold_mem.b.data @= dut_mem.b.data
        gold_mem.r.valid @= dut_mem.r.valid
        gold_mem.r.data @= dut_mem.r.data

        mem_wen0 = m.Bit(name="mem_wen0")
        mem_wdata0 = m.UInt[nasti_params.x_data_bits](name="mem_wdata0")
        mem_wen1 = m.Bit(name="mem_wen1")
        mem_wdata1 = m.UInt[nasti_params.x_data_bits](name="mem_wdata1")
        mem_waddr1 = m.UInt[20](name="mem_waddr1")
        mem.write(
            m.mux([
                dut_mem.w.data.data,
                mem_wdata1
            ], mem_wen1),
            m.mux([
                ((dut_mem.aw.data.addr >> size) +
                 m.zext_to(write_counter.O, nasti_params.x_addr_bits))[:20],
                mem_waddr1
            ], mem_wen1),
            m.enable(mem_wen0 | mem_wen1))

        @m.inline_combinational()
        def mem_fsm():
            dut_mem.w.ready @= False
            dut_mem.aw.ready @= False
            dut_mem.ar.ready @= False

            mem_wen0 @= False

            mem_state.I @= mem_state.O

            if mem_state.O == MemState.IDLE:
                if gold_mem.aw.valid & dut_mem.aw.valid:
                    mem_state.I @= MemState.WRITE
                elif gold_mem.ar.valid & dut_mem.ar.valid:
                    mem_state.I @= MemState.READ
            elif mem_state.O == MemState.WRITE:
                if gold_mem.w.valid & dut_mem.w.valid:
                    mem_wen0 @= True
                    dut_mem.w.ready @= True
                if write_counter.COUT:
                    dut_mem.aw.ready @= True
                    mem_state.I @= MemState.WRITE_ACK
            elif mem_state.O == MemState.WRITE_ACK:
                if gold_mem.b.ready & dut_mem.b.ready:
                    mem_state.I @= MemState.IDLE
            elif mem_state.O == MemState.READ:
                if read_counter.COUT:
                    dut_mem.ar.ready @= True
                    mem_state.I @= MemState.IDLE

        rand_tag = BitVector.random(t_len)
        rand_idx = BitVector.random(s_len)
        rand_off = BitVector.random(b_len) & -4
        rand_data = BitVector[nasti_params.x_data_bits](0)
        for i in range(nasti_params.x_data_bits // 8):
            rand_data |= BitVector[nasti_params.x_data_bits](
                random.randint(0, 0xff) << (8 * i)
            )
        rand_mask = BitVector[x_len // 8](
            random.randint(1, (1 << (x_len // 8)) - 2)
        )

        def test(test_data, tag, idx, off, mask=BitVector[x_len // 8](0)):
            return mask.concat(test_data).concat(tag).concat(idx).concat(off)

        tags = [rand_tag] * 3
        idxs = [rand_idx] * 2
        offs = [rand_off] * 6

        init_addr = []
        _iter = itertools.product(tags, idxs, range(0, data_beats))
        for tag, idx, off in _iter:
            init_addr.append(tag.concat(idx).concat(BitVector[b_len](off)))
        init_data = [rand_data] * len(init_addr)

        test_data = rand_data
        for i in range((b_bits // nasti_params.x_data_bits) - 1):
            test_data = test_data.concat(rand_data)
        test_vec = [
            test(test_data, tags[0], idxs[0], offs[0]),  # 0: read miss
            test(test_data, tags[0], idxs[0], offs[1]),  # 1: read hit
            test(test_data, tags[1], idxs[0], offs[0]),  # 2: read miss
            test(test_data, tags[1], idxs[0], offs[2]),  # 3: read hit
            test(test_data, tags[1], idxs[0], offs[3]),  # 4: read hit
            test(test_data, tags[1], idxs[0], offs[4], rand_mask),  # 5: write hit  # noqa
            test(test_data, tags[1], idxs[0], offs[4]),  # 6: read hit
            test(test_data, tags[2], idxs[0], offs[5]),  # 7: read miss & write back  # noqa
            test(test_data, tags[0], idxs[1], offs[0], rand_mask),  # 8: write miss  # noqa
            test(test_data, tags[0], idxs[1], offs[0]),  # 9: read hit
            test(test_data, tags[0], idxs[1], offs[1]),  # 10: read hit
            test(test_data, tags[1], idxs[1], offs[2], rand_mask),  # 11: write miss & write back  # noqa
            test(test_data, tags[1], idxs[1], offs[3]),  # 12: read hit
            test(test_data, tags[2], idxs[1], offs[4]),  # 13: read write back
            test(test_data, tags[2], idxs[1], offs[5])  # 14: read hit
        ]

        class TestState(m.Enum):
            INIT = 0
            START = 1
            WAIT = 2
            DONE = 3

        state = m.Register(init=TestState.INIT)()
        timeout = m.Register(m.UInt[32])()
        init_m = len(init_addr) - 1
        init_counter = mantle.CounterModM(init_m, init_m.bit_length(),
                                          has_ce=True)
        init_counter.CE @= m.enable(state.O == TestState.INIT)

        test_m = len(test_vec) - 1
        test_counter = mantle.CounterModM(test_m, test_m.bit_length(),
                                          has_ce=True)
        test_counter.CE @= m.enable(state.O == TestState.DONE)
        curr_vec = m.mux(test_vec, test_counter.O)
        mask = (curr_vec >> (b_len + s_len + t_len + b_bits))[:x_len // 8]
        data = (curr_vec >> (b_len + s_len + t_len))[:b_bits]
        tag = (curr_vec >> (b_len + s_len))[:t_len]
        idx = (curr_vec >> b_len)[:s_len]
        off = curr_vec[:b_len]

        dut.cpu.req.data.addr @= m.concat(tag, idx, off)
        # TODO: Is truncating this fine?
        req_data = data[:x_len]
        dut.cpu.req.data.data @= req_data
        dut.cpu.req.data.mask @= mask
        dut.cpu.req.valid @= state.O == TestState.WAIT
        dut.cpu.abort @= 0
        gold_req.data @= dut.cpu.req.data.value()
        gold_req.valid @= state == TestState.START
        gold_resp.ready @= state == TestState.DONE

        mem_waddr1 @= m.mux(init_addr, init_counter.O)[:20]
        mem_wdata1 @= m.mux(init_data, init_counter.O)

        check_resp_data = m.Bit()
        @m.inline_combinational()
        def state_fsm():
            timeout.I @= timeout.O
            mem_wen1 @= m.bit(False)
            check_resp_data @= m.bit(False)
            if state.O == TestState.INIT:
                mem_wen1 @= m.bit(True)
                if init_counter.COUT:
                    state.I @= TestState.START
            elif state.O == TestState.START:
                if gold_req.ready:
                    timeout.I @= m.bits(0, 32)
                    state.I @= TestState.WAIT
            elif state.O == TestState.WAIT:
                timeout.I @= timeout.O + 1
                if dut.cpu.resp.valid & gold_resp.valid:
                    if ~mask.reduce_or():
                        check_resp_data @= m.bit(True)
                    state.I @= TestState.DONE
            elif state.O == TestState.DONE:
                state.I @= TestState.START

        f.assert_immediate(~check_resp_data | (dut.cpu.resp.data.data ==
                                               gold_resp.data.data))
        io.done @= test_counter.COUT

    m.compile("build/CacheDUT", DUT, inline=True)
