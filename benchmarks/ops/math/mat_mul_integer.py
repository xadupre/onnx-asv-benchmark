from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class MatMulInteger(_OperatorBenchmark):
    operator = "MatMulInteger"
    case_name = "test_cc_matmulinteger_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('uint8', 'int8')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
