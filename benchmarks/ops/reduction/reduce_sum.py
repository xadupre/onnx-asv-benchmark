from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ReduceSum(_OperatorBenchmark):
    operator = "ReduceSum"
    case_name = "test_cc_reducesum_default_axes_keepdims_benchmark"
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
