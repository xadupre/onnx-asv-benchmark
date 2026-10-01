# onnx-asv-benchmark

ASV benchmarks comparing inference with:

- ONNX Runtime
- the ONNX reference evaluator
- onnx-light

The benchmark suite groups operators by category:

- `benchmarks/models`: a matrix multiplication with bias and a two-layer MLP
- `benchmarks/maths`: Add and MatMul
- `benchmarks/nn`: Relu

The ASV environment name tracks the pinned dependency versions. ASV's machine
profile records the processor model, architecture, logical CPU count, and
memory with every benchmark result.
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

Before the first run, use ASV's detected CPU description as the machine name
instead of the hostname, and store its machine information:

```bash
processor="$(python - <<'PY'
from asv.machine import Machine, MachineCollection

profile = Machine.get_defaults()
profile["machine"] = profile["cpu"]
MachineCollection.save(profile["machine"], profile)
print(profile["machine"])
PY
)"
```

This creates the ASV machine profile (in `~/.asv-machine.json` by default).
It is required even when the processor is visible to the operating system,
including under WSL. Keep `processor` set in the shell for the following
commands so ASV selects that profile rather than the hostname.

Run the complete comparison in the versioned environment:

```bash
asv run --machine "$processor"
```

Run against the currently active Python environment:

```bash
asv run --environment existing --machine "$processor"
```

For a quick smoke test:

```bash
asv run --quick --machine "$processor"
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
of [xadupre/cache_data](https://github.com/xadupre/cache_data), then trigger
the GitHub Pages deployment:

```bash
python tools/publish_results.py
```

The command clones `cache_data`, merges the local ASV results into the shared
subdirectory, commits any changes, and pushes them to its `main` branch. It
then starts the `Publish benchmark pages` workflow, which builds the ASV site
from the raw results and deploys it to
[xadupre.github.io/onnx-asv-benchmark](https://xadupre.github.io/onnx-asv-benchmark/).

Git credentials with write access to `xadupre/cache_data` and an authenticated
[GitHub CLI](https://cli.github.com/) with Actions access to this repository
must be configured. Pass `--skip-pages` to publish only the raw results.
