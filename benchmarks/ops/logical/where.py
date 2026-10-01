from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Where(_OperatorBenchmark):
    operator = "Where"
    case_name = "test_where_example_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
