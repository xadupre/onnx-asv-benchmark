from benchmarks._operator import OperatorBenchmark


class RMSNormalization(OperatorBenchmark):
    operator = "RMSNormalization"
    case_name = "test_cc_rms_normalization_2d_axis0_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
