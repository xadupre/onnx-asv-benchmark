from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LayerNormalization(_OperatorBenchmark):
    operator = "LayerNormalization"
    case_name = "test_cc_layer_normalization_2d_axis0_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
