from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class CumSum(_OperatorBenchmark):
    operator = "CumSum"
    case_name = "test_cc_cumsum_benchmark"
    case_mode = "BENCHMARK"
    dtypes = (
        "float64",
        "uint32",
        "uint64",
        "int32",
        "int64",
        "float16",
        "float32",
        "bfloat16",
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
