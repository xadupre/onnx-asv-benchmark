from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Resize(_OperatorBenchmark):
    operator = "Resize"
    case_name = "test_cc_resize_upsample_scales_nearest_asymmetric_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
