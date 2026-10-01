from benchmarks._operator import OperatorBenchmark


class Add(OperatorBenchmark):
    operator = "Add"
    case_name = "test_cc_add_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
