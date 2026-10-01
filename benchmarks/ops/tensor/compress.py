from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Compress(_OperatorBenchmark):
    operator = "Compress"
    case_name = "test_cc_compress_no_axis_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
