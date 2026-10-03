from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ConvInteger(_OperatorBenchmark):
    operator = "ConvInteger"
    case_name = "test_cc_basic_convinteger_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('uint8', 'int8')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
