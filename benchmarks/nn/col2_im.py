from benchmarks._operator import OperatorBenchmark


class Col2Im(OperatorBenchmark):
    operator = "Col2Im"
    case_name = "test_cc_col2im_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
