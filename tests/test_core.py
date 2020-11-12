import pytest
from hwtypes import BitVector
import magma as m


class SimpleTests:
    tests = ["rv32ui-p-simple"]
    maxcycles = 150000


class ISATests:
    tests = []
    for test in ["simple", "add", "addi", "auipc", "and", "andi",
                 "sb", "sh", "sw", "lb", "lbu", "lh", "lhu", "lui", "lw",
                 "beq", "bge", "bgeu", "blt", "bltu", "bne", "j", "jal",
                 "jalr", "or", "ori", "sll", "slli", "slt", "slti", "sra",
                 "srai", "sub", "xor", "xori"]:
        # TODO: "fence_i" (also todo in chisel riscv-mini)
        tests.append(f"rv32ui-p-{test}")
    for test in ["sbreak", "scall", "illegal", "ma_fetch", "ma_addr", "csr"]:
        # TODO: "timer" (also todo in chisel riscv-mini)
        tests.append(f"rv32mi-p-{test}")
        maxcycles = 15000


class BmarkTests:
    tests = ["median.riscv", "multiply.riscv", "qsort.riscv", "towers.riscv",
             "vvadd.riscv"]
    maxcycles = 1500000


class LargeBmarkTests:
    tests = ["median.riscv-large", "multiply.riscv-large", "qsort.riscv-large",
             "towers.riscv-large", "vvadd.riscv-large"]
    maxcycles = 5000000


def parse_nibble(hex_str):
    hex_val = ord(hex_str)
    if hex_val >= ord('a'):
        return hex_val - ord('a') + 10
    return hex_val - ord('0')


def load_mem(lines, chunk):
    insts = []
    for line in lines:
        assert len(line) % (chunk // 4) == 0
        for i in range(len(line) - (chunk // 4), -1, -(chunk // 4)):
            inst = 0
            for j in range(0, chunk // 4):
                inst |= (parse_nibble(line[i + j]) <<
                         (4 * ((chunk // 4) - (j + 1))))
            insts.append(BitVector[chunk](inst))
    chunks = []
    for i in range(0, len(insts), 1 << 8):
        chunks.append(insts[i:i + 1 << 8])
    return chunks


@pytest.mark.parametrize('test', [SimpleTests, ISATests, BmarkTests])
def test_core(test):
    for t in test.tests:
        with open(f'tests/resources/{t}.hex', 'r') as f:
            contents = [line.rstrip() for line in f.readlines()]
            loadmem = load_mem(contents, 32)
