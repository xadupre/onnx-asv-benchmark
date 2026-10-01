from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ScatterND(_OperatorBenchmark):
    operator = "ScatterND"
    case_name = "test_cc_scatternd_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
