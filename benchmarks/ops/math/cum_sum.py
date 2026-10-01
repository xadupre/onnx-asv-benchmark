from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class CumSum(_OperatorBenchmark):
    operator = "CumSum"
    case_name = "test_cc_cumsum_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
