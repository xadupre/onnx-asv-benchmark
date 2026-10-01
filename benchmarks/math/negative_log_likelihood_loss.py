from benchmarks._operator import OperatorBenchmark


class NegativeLogLikelihoodLoss(OperatorBenchmark):
    operator = "NegativeLogLikelihoodLoss"
    case_name = "test_cc_negative_log_likelihood_loss_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
