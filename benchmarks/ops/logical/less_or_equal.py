from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LessOrEqual(_OperatorBenchmark):
    operator = "LessOrEqual"
    case_name = "test_cc_less_or_equal_benchmark"
    case_mode = "BENCHMARK"
    dtypes = (
        "float32",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float64",
        "bfloat16",
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
