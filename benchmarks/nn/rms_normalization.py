from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class RMSNormalization(_OperatorBenchmark):
    operator = "RMSNormalization"
    case_name = "test_cc_rms_normalization_2d_axis0_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
