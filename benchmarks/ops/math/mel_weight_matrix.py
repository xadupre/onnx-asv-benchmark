from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class MelWeightMatrix(_OperatorBenchmark):
    operator = "MelWeightMatrix"
    case_name = "test_cc_melweightmatrix_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('int32', 'int64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
