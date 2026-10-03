from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Split(_OperatorBenchmark):
    operator = "Split"
    case_name = "test_cc_split_equal_parts_1d_opset13_benchmark"
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
        "bfloat16",
        "float16",
        "float64",
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
