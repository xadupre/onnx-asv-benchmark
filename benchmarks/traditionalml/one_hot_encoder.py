from benchmarks._operator import OperatorBenchmark


class OneHotEncoder(OperatorBenchmark):
    operator = "OneHotEncoder"
    case_name = "test_cc_one_hot_encoder_int64_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
