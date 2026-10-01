from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Split(_OperatorBenchmark):
    operator = "Split"
    case_name = "test_cc_split_equal_parts_1d_opset13_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
