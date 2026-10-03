from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class MatMul(_OperatorBenchmark):
    operator = "MatMul"
    case_name = "test_cc_matmul_benchmark"
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
