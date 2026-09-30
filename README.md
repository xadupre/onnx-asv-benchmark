# onnx-asv-benchmark

ASV benchmarks comparing inference with:

- ONNX Runtime
- the ONNX reference evaluator
- onnx-light

The benchmark suite follows ONNX's separation between model and node cases:

- `benchmarks/models`: a matrix multiplication with bias and a two-layer MLP
- `benchmarks/operators`: Add, MatMul, and Relu in isolation

The ASV environment name tracks the pinned dependency versions.
`benchmarks/versions.py` also records the version reported by every runtime.

## Setup

Install ASV:

```bash
python -m pip install asv
```

ASV creates a Python 3.12 virtual environment and installs the pinned NumPy,
ONNX, and ONNX Runtime releases from PyPI. It installs onnx-light 0.1.28 from
the wheel published on the
[onnx-light release page](https://github.com/xadupre/onnx-light/releases/tag/0.1.28).

## Run

Run the complete comparison in the versioned environment:

```bash
asv run
```

For a quick smoke test:

```bash
asv run --quick
```
