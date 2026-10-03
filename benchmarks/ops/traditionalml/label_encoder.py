from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LabelEncoder(_OperatorBenchmark):
    operator = "LabelEncoder"
    case_name = "test_cc_label_encoder_int64_to_float_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('int64', 'float32', 'int32', 'int16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
