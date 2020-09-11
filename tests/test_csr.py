import magma as m
from riscv_mini.csr_gen import CSRGen


def test_csr():
    m.compile("build/CSR", CSRGen(32))
