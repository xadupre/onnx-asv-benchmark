from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class QLinearMatMul(_OperatorBenchmark):
    operator = "QLinearMatMul"
    case_name = "test_cc_qlinearmatmul_2D_uint8_float32_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('uint8', 'int8')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
