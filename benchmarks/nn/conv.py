from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Conv(_OperatorBenchmark):
    operator = "Conv"
    case_name = "test_cc_basic_conv_without_padding_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
