from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ReduceMax(_OperatorBenchmark):
    operator = "ReduceMax"
    case_name = "test_cc_reducemax_default_axes_keepdims_benchmark"
    case_mode = "BENCHMARK"
    dtypes = (
        "float32",
        "uint32",
        "uint64",
        "int32",
        "int64",
        "float16",
        "float64",
        "bfloat16",
        "uint8",
        "int8",
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
