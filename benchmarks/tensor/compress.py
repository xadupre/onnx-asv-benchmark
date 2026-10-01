from benchmarks._operator import OperatorBenchmark


class Compress(OperatorBenchmark):
    operator = "Compress"
    case_name = "test_cc_compress_no_axis_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
