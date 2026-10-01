from benchmarks._operator import OperatorBenchmark


class Clip(OperatorBenchmark):
    operator = "Clip"
    case_name = "test_cc_clip_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
