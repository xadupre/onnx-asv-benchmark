from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class Upsample(_OperatorBenchmark):
    operator = "Upsample"
    case_name = "test_cc_upsample_nearest_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
