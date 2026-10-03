from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Resize(_OperatorBenchmark):
    operator = "Resize"
    case_name = "test_cc_resize_upsample_scales_nearest_asymmetric_benchmark"
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
