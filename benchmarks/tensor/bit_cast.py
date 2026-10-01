from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class BitCast(_OperatorBenchmark):
    operator = "BitCast"
    case_name = "test_cc_bitcast_float_to_int32_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
