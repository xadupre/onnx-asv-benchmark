from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LayerNormalization(_OperatorBenchmark):
    operator = "LayerNormalization"
    case_name = "test_cc_layer_normalization_2d_axis0_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float16', 'float64', 'bfloat16')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
