from benchmarks._operator import OperatorBenchmark


class Slice(OperatorBenchmark):
    operator = "Slice"
    case_name = "test_cc_slice_axes_steps_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
