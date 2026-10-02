from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class DeformConv(_OperatorBenchmark):
    operator = "DeformConv"
    case_name = "test_cc_basic_deform_conv_without_padding_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
