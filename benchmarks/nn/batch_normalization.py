from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class BatchNormalization(_OperatorBenchmark):
    operator = "BatchNormalization"
    case_name = "test_cc_batchnorm_example_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
