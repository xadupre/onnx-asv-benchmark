from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Cast(_OperatorBenchmark):
    operator = "Cast"
    case_name = "test_cc_cast_FLOAT_to_DOUBLE_benchmark"
    case_mode = "BENCHMARK"
    dtypes = (
        "float32",
        "float16",
        "float64",
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
