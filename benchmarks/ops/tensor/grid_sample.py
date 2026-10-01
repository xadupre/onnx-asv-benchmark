from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class GridSample(_OperatorBenchmark):
    operator = "GridSample"
    case_name = "test_cc_gridsample_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-light")
