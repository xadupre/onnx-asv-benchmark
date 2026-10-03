from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class NegativeLogLikelihoodLoss(_OperatorBenchmark):
    operator = "NegativeLogLikelihoodLoss"
    case_name = "test_cc_negative_log_likelihood_loss_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float16', 'float64')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
