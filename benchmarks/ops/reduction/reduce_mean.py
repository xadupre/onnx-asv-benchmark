from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ReduceMean(_OperatorBenchmark):
    operator = "ReduceMean"
    case_name = "test_cc_reducemean_default_axes_keepdims_benchmark"
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
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
