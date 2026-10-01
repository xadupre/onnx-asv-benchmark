from benchmarks._operator import OperatorBenchmark


class CenterCropPad(OperatorBenchmark):
    operator = "CenterCropPad"
    case_name = "test_cc_center_crop_pad_crop_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
