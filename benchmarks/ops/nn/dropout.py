from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Dropout(_OperatorBenchmark):
    operator = "Dropout"
    case_name = "test_cc_dropout_default_inference_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
