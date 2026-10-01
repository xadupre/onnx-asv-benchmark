from benchmarks._operator import OperatorBenchmark


class SoftmaxCrossEntropyLoss(OperatorBenchmark):
    operator = "SoftmaxCrossEntropyLoss"
    case_name = "test_cc_softmax_cross_entropy_loss_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
