from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class LSTM(_OperatorBenchmark):
    operator = "LSTM"
    case_name = "test_cc_lstm_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-light",)
