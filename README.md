# onnx-asv-benchmark

ASV benchmarks comparing inference with:

- ONNX Runtime
- the ONNX reference evaluator
- onnx-light
- onnx-light-cpu

Operator benchmarks live under `benchmarks/ops` and use the same category
directories as onnx-light. Model benchmarks live under `benchmarks/models`,
split between `llm` and `dummies`.

Builder benchmarks live under `benchmarks/builder`. They cover ONNX graph
construction, I/O, and pattern fusion. The pattern-fusion benchmark compares
onnx-light and onnxscript construction and rewriting for 50, 100, 250, and 500
repeated blocks. Each block contains four fusion patterns and twelve ONNX
nodes; construction and fusion are timed separately.

The ASV environment name tracks the pinned dependency versions. Graph
observation tooltips show the ONNX, onnxscript, and ir-py versions. ASV's machine
profile records the processor model, architecture, logical CPU count,
available SIMD instruction sets, and memory with every benchmark result. The
published home page summarizes the logical CPU count and instruction sets for
every benchmark processor.
All models are created and checked with onnx-light, then serialized for the
ONNX reference evaluator and ONNX Runtime.

## Setup

Install ASV:

```bash
python -m pip install asv
```

ASV creates a Python 3.12 virtual environment and installs the pinned NumPy,
ONNX, and ONNX Runtime releases from PyPI. It installs onnx-light 0.1.30 from
the wheel published on the
[onnx-light release page](https://github.com/xadupre/onnx-light/releases/tag/0.1.30)
and onnx-light-cpu 0.1.20 from its
[release page](https://github.com/xadupre/onnx-light-cpu/releases/tag/0.1.20).

## Run

Before the first run, set up the processor-and-core-count-named ASV machine
profile:

```bash
machine="$(python tools/setup_machine.py)"
```

The script creates or corrects the profile in `~/.asv-machine.json`, reports
what it did on stderr, and prints only the machine name on stdout for shell
capture. The processor and available logical core count identify the machine
in results, so restricted workers are not merged with workers using all cores,
without exposing the hostname. It can be run again without changing a correct
profile. To verify the current setup, use `python tools/setup_machine.py
--check`.

Run the complete comparison in the versioned environment:

```bash
asv run --machine "$machine"
```

Run against the currently active Python environment:

```bash
python tools/run_asv.py
```

This command creates or updates the machine profile, selects the current
Python interpreter, and preserves `PYTHONPATH` for dependencies used directly
from source checkouts. Use `--quick` for a smoke test:

```bash
python tools/run_asv.py --quick
```

Use `--bench` to select benchmarks, optionally at a specific revision:

```bash
python tools/run_asv.py --bench MatMul main^!
```

For a quick smoke test in the versioned environment:

```bash
asv run --quick --machine "$machine"
```

Each benchmark has its own `number` of timed calls per sample, from one for
MatMul to five for Relu. Every benchmark has a 10-second timeout.

Every result records the complete input tensor shapes as an ASV `shape`
parameter, alongside `dtype` and `backend` where applicable. Operator
benchmarks include every floating-point and integer dtype admitted by their ONNX
schema while preserving the dtype of indices and other independently typed inputs. Multi-input
operators include each input name and shape; scalar and input-free cases are
identified explicitly. Tiny-LLM and Qwen2 record the prefill, decode, cache,
and generation shapes. Qwen2 also records `Qwen/Qwen2-0.5B` as its `model`
parameter and uses `Qwen2-0.5B` in graph titles.

`builder/load/onnx_io`, `builder/save/onnx_io`, `builder/serialize/onnx_io`,
and `builder/parse/onnx_io` benchmark the 42 load, save, serialize, parse, and
standalone C++ cases in onnx-light's `plot_onnx_time.py`. They use that
example's 40-Gemm float32 model with 2048-wide weights; model creation and
fixture file preparation happen outside the timed call. Install `onnx-ir` to
run the `ir-py` cases. The C++ cases require the onnx-light example executables
(`load_onnx_time`, `load_onnx_light_time`, `save_onnx_light_time`) on `PATH`
or in their onnx-light build directories; in CI, set `CICPP=1` to enable
executable discovery. These C++ results are tracked seconds per operation
reported by the executables, not Python subprocess launch time.

To check that every benchmark runs on all three backends in an environment
with the benchmark dependencies installed:

```bash
python -m pytest tests
```

The operator benchmarks mirror the `onnx-light` kernel categories and contain
one module for each registered operator. Regenerate them from a neighboring
`onnx-light` source checkout after its kernel registry changes:

```bash
PYTHONPATH=../onnx-light python tools/generate_operator_benchmarks.py ../onnx-light
```

Each generated benchmark uses the corresponding native `onnx-light` backend
benchmark case and declares ONNX Runtime, ONNX Reference, onnx-light, and
onnx-light-cpu unconditionally. Unsupported or numerically inconsistent
backends remain visible as failed measurements instead of being excluded.
onnx-light-cpu kernels are registered only on that benchmark session, while
operators without one fall back to onnx-light. This keeps the regular
onnx-light measurements unchanged.

The model benchmarks also include one-layer `arnir0/Tiny-LLM` and
`Qwen/Qwen2-0.5B` configurations based on `mbext` fast tests. Each benchmark
creates a deterministic random Hugging Face model and uses `mbext` to generate
the ONNX model before timing prefill, single-token decode with a 128-token KV
cache, and generation from a text prompt. Prefill and decode include ONNX
Runtime, ONNX Reference, onnx-light, and onnx-light-cpu. Generation includes
ONNX Runtime GenAI, ONNX Reference, onnx-light, and onnx-light-cpu; the
lightweight backends use `ReferenceEvaluator.generate`, while ONNX Reference
uses the equivalent greedy token loop. Both configurations use hidden size
512, intermediate size 1376, eight attention heads, four key/value heads, and
a 32,000-token vocabulary. All three scenarios cover FP32, FP16, BF16, INT8,
INT4, and INT2 models generated by `mbext` on CPU. BF16 is skipped only for
ONNX Runtime and ONNX Runtime GenAI, which do not provide the required CPU
kernels.

The `builder/builder/graph_builder` benchmark compares onnx-light and
onnxscript construction of the same dynamic attention graph at 100, 200, 500,
1000, and 2000 nodes. It times model finalization with and without in-memory
serialization; each model contains four 8 MB initializers (32 MB total), rather
than the example's 1.5 GB payload, to keep scheduled runs within runner memory.
Input generation and model execution are excluded from the timings.

The weekly `GenAI compatibility` workflow runs both generation scenarios with
ONNX Runtime GenAI and onnx-light-cpu. Its matrix tests the latest onnx-light
and onnx-light-cpu releases together, then builds and tests both projects from
their `main` branches. The same check can be run in an installed environment
with:

```bash
python tools/check_genai.py
```

## Publish results

Publish the raw `.asv/results` data as independent shards under the
`onnx-asv-benchmark` subdirectory of
[xadupre/cache_data](https://github.com/xadupre/cache_data):

```bash
python tools/publish_results.py --shard models/llm/tiny_llm
python tools/publish_results.py --shard ops/math
```

Model shards use `models/<group>/<module>`; operator shards use
`ops/<category>`. Omitting `--shard` publishes every shard found in the local
results. The first publication migrates legacy flat results, including the
`xadupre2025` directory, to a processor-and-core-count-named layout.

When the site data is generated, historical benchmark names are normalized to
the same hierarchy. The HTML navigation therefore groups operators under
`ops/<category>` and models under `models/<group>/<module>`, including results
recorded before the source tree was reorganized.

The command clones `cache_data`, merges only the requested local results,
rebases concurrent shard updates, commits any changes, and pushes them to its
`main` branch. The `Publish benchmark data` workflow merges all shards after
each daily benchmark bucket and stores the generated ASV JSON under
`onnx-asv-benchmark/site-data` in `cache_data`. The static HTML deployed to
[xadupre.github.io/onnx-asv-benchmark](https://xadupre.github.io/onnx-asv-benchmark/)
loads `info.json`, `index.json`, and graph data directly from that directory
through `raw.githubusercontent.com`. New benchmark data therefore appears
without rebuilding or redeploying the HTML. Published graphs use dates,
rather than commit positions, on the time axis.

Git credentials with write access to `xadupre/cache_data` and an authenticated
[GitHub CLI](https://cli.github.com/) with Actions access to this repository
must be configured. Pass `--publish-pages` to rebuild the static HTML after
changing its presentation; the workflow can also be started manually from
GitHub Actions.

The `Twice-weekly benchmark shards` workflow assigns every operator category
and model module to one of seven schedule buckets. Each bucket runs twice per
week, with its executions spaced approximately three and a half days apart,
and every shard finishes independently. Configure a `CACHE_DATA_TOKEN` Actions
secret with write access to `xadupre/cache_data`. Shard completion does not
trigger a Pages deployment; it refreshes only the remote JSON data. The static
Pages workflow runs when its presentation files change or when started
manually.
