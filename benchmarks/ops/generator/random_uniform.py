from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class RandomUniform(_OperatorBenchmark):
    operator = "RandomUniform"
    case_name = "test_cc_randomuniform_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
