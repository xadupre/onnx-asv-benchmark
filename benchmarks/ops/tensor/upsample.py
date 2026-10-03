from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Upsample(_OperatorBenchmark):
    operator = "Upsample"
    case_name = "test_cc_upsample_nearest_benchmark"
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
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
