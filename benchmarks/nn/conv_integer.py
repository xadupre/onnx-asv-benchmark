from benchmarks._operator import OperatorBenchmark


class ConvInteger(OperatorBenchmark):
    operator = "ConvInteger"
    case_name = "test_cc_basic_convinteger_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
