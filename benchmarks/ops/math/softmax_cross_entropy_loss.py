from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class SoftmaxCrossEntropyLoss(_OperatorBenchmark):
    operator = "SoftmaxCrossEntropyLoss"
    case_name = "test_cc_softmax_cross_entropy_loss_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('float32', 'float16', 'float64', 'bfloat16')
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
