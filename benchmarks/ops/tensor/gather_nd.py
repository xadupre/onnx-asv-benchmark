from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class GatherND(_OperatorBenchmark):
    operator = "GatherND"
    case_name = "test_cc_gathernd_example_int32_benchmark"
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
        "bfloat16",
        "float16",
        "float32",
        "float64",
    )
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
