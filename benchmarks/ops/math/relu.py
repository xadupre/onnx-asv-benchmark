from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Relu(_OperatorBenchmark):
    operator = "Relu"
    case_name = "test_cc_relu_benchmark"
    case_mode = "BENCHMARK"
    dtypes = (
        "float32",
        "int32",
        "int8",
        "int16",
        "int64",
        "float16",
        "float64",
        "bfloat16",
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
