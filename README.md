# magma-riscv-mini
Magma port of https://github.com/ucb-bar/riscv-mini

## Dependencies
### Ubuntu
```
sudo apt install verilator libgmp-dev libmpfr-dev libmpc-dev
```
### MacOS
```
brew install verilator gmp mpfr libmpc
```

## Test
```
pip install -e .
pytest tests
```
