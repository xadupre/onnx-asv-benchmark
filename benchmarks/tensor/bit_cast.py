from benchmarks._operator import OperatorBenchmark


class BitCast(OperatorBenchmark):
    operator = "BitCast"
    case_name = "test_cc_bitcast_float_to_int32_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
