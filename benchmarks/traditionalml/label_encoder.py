from benchmarks._operator import OperatorBenchmark


class LabelEncoder(OperatorBenchmark):
    operator = "LabelEncoder"
    case_name = "test_cc_label_encoder_int64_to_float_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
