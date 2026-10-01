from benchmarks._operator import OperatorBenchmark


class InstanceNormalization(OperatorBenchmark):
    operator = "InstanceNormalization"
    case_name = "test_cc_instancenorm_example_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
