from benchmarks._operator import OperatorBenchmark


class Dropout(OperatorBenchmark):
    operator = "Dropout"
    case_name = "test_cc_dropout_default_inference_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
