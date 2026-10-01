from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class OneHotEncoder(_OperatorBenchmark):
    operator = "OneHotEncoder"
    case_name = "test_cc_one_hot_encoder_int64_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
