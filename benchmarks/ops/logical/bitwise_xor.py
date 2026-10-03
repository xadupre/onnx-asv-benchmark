from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class BitwiseXor(_OperatorBenchmark):
    operator = "BitwiseXor"
    case_name = "test_cc_bitwise_xor_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('int32', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
