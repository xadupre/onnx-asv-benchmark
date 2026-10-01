from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class CumProd(_OperatorBenchmark):
    operator = "CumProd"
    case_name = "test_cc_cumprod_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
