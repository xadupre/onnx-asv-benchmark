from benchmarks._operator import OperatorBenchmark


class ImageDecoder(OperatorBenchmark):
    operator = "ImageDecoder"
    case_name = "test_cc_image_decoder_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnx-reference", "onnx-light")
