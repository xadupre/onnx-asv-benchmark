from benchmarks._operator import OperatorBenchmark


class Det(OperatorBenchmark):
    operator = "Det"
    case_name = "test_cc_det_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
