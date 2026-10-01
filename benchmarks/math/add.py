from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Add(_OperatorBenchmark):
    operator = "Add"
    case_name = "test_cc_add_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
