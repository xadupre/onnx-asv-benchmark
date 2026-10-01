from benchmarks._operator import OperatorBenchmark


class Upsample(OperatorBenchmark):
    operator = "Upsample"
    case_name = "test_cc_upsample_nearest_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
