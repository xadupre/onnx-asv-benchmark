# onnx-asv-benchmark

ASV benchmarks comparing inference with:

- ONNX Runtime
- the ONNX reference evaluator
- onnx-light

The benchmark suite follows ONNX's separation between model and node cases:

- `benchmarks/models`: a matrix multiplication with bias and a two-layer MLP
- `benchmarks/operators`: Add, MatMul, and Relu in isolation

`benchmarks/versions.py` records the installed version of every runtime in each
ASV result set.

## Setup

The benchmarks use ASV's existing environment. Install ASV, ONNX, and ONNX
Runtime from PyPI:

```bash
python -m pip install -r requirements.txt
```

Then install the onnx-light wheel for your Python version and platform from the
[onnx-light 0.1.28 release page](https://github.com/xadupre/onnx-light/releases/tag/0.1.28).
For CPython 3.12 on x86-64 Linux:

```bash
python -m pip install https://github.com/xadupre/onnx-light/releases/download/0.1.28/onnx_light-0.1.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

## Run

Run the complete comparison in the active environment:

```bash
asv run --environment existing
```

For a quick smoke test:

```bash
asv run --environment existing --quick
```
