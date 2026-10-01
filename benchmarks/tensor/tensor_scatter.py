from benchmarks._operator import OperatorBenchmark


class TensorScatter(OperatorBenchmark):
    operator = "TensorScatter"
    case_name = "test_cc_tensorscatter_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
