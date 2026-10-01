from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Det(_OperatorBenchmark):
    operator = "Det"
    case_name = "test_cc_det_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
