# onnx-asv-benchmark

ASV benchmarks comparing inference with:

- ONNX Runtime
- the ONNX reference evaluator
- onnx-light

The benchmark suite groups operators by category:

- `benchmarks/models`: a matrix multiplication with bias and a two-layer MLP
- `benchmarks/maths`: Add and MatMul
- `benchmarks/nn`: Relu

The ASV environment name tracks the pinned dependency versions.
`benchmarks/versions.py` records the installed NumPy and runtime versions.
`benchmarks/machine.py` records the processor model, architecture, and logical
CPU count with every benchmark result.
All models are created and checked with onnx-light, then serialized for the
ONNX reference evaluator and ONNX Runtime.

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

Run against the currently active Python environment:

```bash
asv run --environment existing
```

For a quick smoke test:

```bash
asv run --quick
```
