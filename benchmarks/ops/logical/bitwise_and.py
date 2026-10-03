from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class BitwiseAnd(_OperatorBenchmark):
    operator = "BitwiseAnd"
    case_name = "test_cc_bitwise_and_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('int32', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
