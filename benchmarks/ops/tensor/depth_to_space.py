from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class DepthToSpace(_OperatorBenchmark):
    operator = "DepthToSpace"
    case_name = "test_cc_depthtospace_dcr_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
