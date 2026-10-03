from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class OneHot(_OperatorBenchmark):
    operator = "OneHot"
    case_name = "test_onehot_without_axis_benchmark"
    case_mode = "BENCHMARK"
    dtypes = (
        "int32",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int64",
        "float16",
        "float32",
        "float64",
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
