from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class ImageDecoder(_OperatorBenchmark):
    operator = "ImageDecoder"
    case_name = "test_cc_image_decoder_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('uint8',)
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
