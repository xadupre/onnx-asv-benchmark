from benchmarks._operator import OperatorBenchmark


class TopK(OperatorBenchmark):
    operator = "TopK"
    case_name = "test_cc_top_k_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
