from benchmarks._operator import OperatorBenchmark


class Where(OperatorBenchmark):
    operator = "Where"
    case_name = "test_where_example_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
