from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Clip(_OperatorBenchmark):
    operator = "Clip"
    case_name = "test_cc_clip_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
