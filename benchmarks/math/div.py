from benchmarks._operator import OperatorBenchmark


class Div(OperatorBenchmark):
    operator = "Div"
    case_name = "test_cc_div_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
