from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Col2Im(_OperatorBenchmark):
    operator = "Col2Im"
    case_name = "test_cc_col2im_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
