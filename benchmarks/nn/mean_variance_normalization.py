from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class MeanVarianceNormalization(_OperatorBenchmark):
    operator = "MeanVarianceNormalization"
    case_name = "test_cc_mvn_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
