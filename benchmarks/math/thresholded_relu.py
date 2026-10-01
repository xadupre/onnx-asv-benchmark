from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ThresholdedRelu(_OperatorBenchmark):
    operator = "ThresholdedRelu"
    case_name = "test_cc_thresholdedrelu_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
