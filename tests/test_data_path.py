import magma as m

from riscv_mini.data_path import Datapath


def test_datapath():
    m.compile("build/Datapath", Datapath(32))
