# magma-riscv-mini
Magma port of https://github.com/ucb-bar/riscv-mini

## Dependencies
### Ubuntu
```
sudo apt install verilator libgmp-dev libmpfr-dev libmpc-dev
pip install pytest magma-lang mantle fault
```
### MacOS
```
brew install verilator gmp mpfr libmpc
pip install pytest magma-lang mantle fault
```

## Test
```
pytest tests
```
