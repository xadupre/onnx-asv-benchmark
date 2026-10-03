from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Concat(_OperatorBenchmark):
    operator = "Concat"
    case_name = "test_cc_concat_1d_axis_0_benchmark"
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
