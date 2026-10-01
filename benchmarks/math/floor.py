from benchmarks._operator import OperatorBenchmark


class Floor(OperatorBenchmark):
    operator = "Floor"
    case_name = "test_cc_floor_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
