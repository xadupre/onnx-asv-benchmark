from benchmarks._operator import OperatorBenchmark


class DepthToSpace(OperatorBenchmark):
    operator = "DepthToSpace"
    case_name = "test_cc_depthtospace_dcr_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
