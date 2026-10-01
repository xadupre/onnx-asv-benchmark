from benchmarks._operator import OperatorBenchmark


class MelWeightMatrix(OperatorBenchmark):
    operator = "MelWeightMatrix"
    case_name = "test_cc_melweightmatrix_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
