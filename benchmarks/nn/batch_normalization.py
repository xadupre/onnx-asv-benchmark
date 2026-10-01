from benchmarks._operator import OperatorBenchmark


class BatchNormalization(OperatorBenchmark):
    operator = "BatchNormalization"
    case_name = "test_cc_batchnorm_example_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
