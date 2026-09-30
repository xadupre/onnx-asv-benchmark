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

Before the first run on a machine, let ASV detect and store its machine
information:

```bash
asv machine --yes
```

This creates the ASV machine profile (in `~/.asv-machine.json` by default).
It is required even when the processor is visible to the operating system,
including under WSL.

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

Each benchmark has its own `number` of timed calls per sample, from one for
MatMul to five for Relu. Every benchmark has a 10-second timeout.

To check that every benchmark runs on all three backends in an environment
with the benchmark dependencies installed:

```bash
python -m pytest tests
```

## Publish results

Publish the raw `.asv/results` data to the `onnx-asv-benchmark` subdirectory
of [xadupre/cache_data](https://github.com/xadupre/cache_data):

```bash
python tools/publish_results.py
```

The command clones `cache_data`, merges the local ASV results into the shared
subdirectory, commits any changes, and pushes them to its `main` branch. Git
credentials with write access to `xadupre/cache_data` must be configured.
