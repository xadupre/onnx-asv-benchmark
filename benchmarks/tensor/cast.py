from benchmarks._operator import OperatorBenchmark


class Cast(OperatorBenchmark):
    operator = "Cast"
    case_name = "test_cc_cast_FLOAT_to_DOUBLE_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
