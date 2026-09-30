# onnx-asv-benchmark

ASV benchmarks comparing inference on two ONNX models with:

- ONNX Runtime
- the ONNX reference evaluator
- onnx-light
- onnx-light with onnx-light-cpu kernels

The benchmark models are generated in memory. They cover a matrix multiplication
with a bias and a two-layer MLP.

## Setup

The benchmarks use ASV's existing environment so that onnx-light-cpu can be
built against the same onnx-light shared library used at runtime. Install ASV
and the standard Python dependencies:

```bash
python -m pip install asv numpy onnx onnxruntime
```

Build both source checkouts in place, with the onnx-light-cpu integration
enabled:

```bash
cd ../onnx-light
python setup.py build_ext --inplace
cd ../onnx-light-cpu
PYTHONPATH=../onnx-light python setup.py build_ext --inplace --onnx-light-source
cd ../onnx-asv-benchmark
export PYTHONPATH="../onnx-light:../onnx-light-cpu:${PYTHONPATH}"
```

Verify that the optimized kernels are available:

```bash
python -c "from onnx_light_cpu import has_cpu_kernels, register_kernels_global; assert has_cpu_kernels(); register_kernels_global()"
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
