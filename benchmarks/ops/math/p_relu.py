from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class PRelu(_OperatorBenchmark):
    operator = "PRelu"
    case_name = "test_cc_prelu_benchmark"
    case_mode = "BENCHMARK"
    dtypes = (
        "float32",
        "bfloat16",
        "float16",
        "float64",
        "uint32",
        "uint64",
        "int32",
        "int64",
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
