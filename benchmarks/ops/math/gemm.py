from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Gemm(_OperatorBenchmark):
    operator = "Gemm"
    case_name = "test_cc_gemm_benchmark"
    case_mode = "BENCHMARK"
    dtypes = (
        "float32",
        "float16",
        "float64",
        "uint32",
        "uint64",
        "int32",
        "int64",
        "bfloat16",
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
