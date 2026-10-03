from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class CastLike(_OperatorBenchmark):
    operator = "CastLike"
    case_name = "test_cc_castlike_FLOAT_to_DOUBLE_benchmark"
    case_mode = "BENCHMARK"
    dtypes = (
        "float64",
        "float16",
        "float32",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bfloat16",
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
