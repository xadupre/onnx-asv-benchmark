from benchmarks._operator import OperatorBenchmark


class LayerNormalization(OperatorBenchmark):
    operator = "LayerNormalization"
    case_name = "test_cc_layer_normalization_2d_axis0_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
