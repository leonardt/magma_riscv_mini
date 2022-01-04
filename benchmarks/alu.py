import timeit
from riscv_mini.alu import ALUSimple, ALUArea

simple = timeit.Timer(lambda: ALUSimple(32)).timeit(number=2)
print(simple)

area = timeit.Timer(lambda: ALUArea(32)).timeit(number=2)
print(area)
