from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class RNN(_OperatorBenchmark):
    operator = "RNN"
    case_name = "test_cc_rnn_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-light",)
