from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Div(_OperatorBenchmark):
    operator = "Div"
    case_name = "test_cc_div_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
