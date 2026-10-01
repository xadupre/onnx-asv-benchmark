from benchmarks._operator import OperatorBenchmark


class DeformConv(OperatorBenchmark):
    operator = "DeformConv"
    case_name = "test_cc_basic_deform_conv_without_padding_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-light")
