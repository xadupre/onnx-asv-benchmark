from benchmarks._operator import OperatorBenchmark


class CategoryMapper(OperatorBenchmark):
    operator = "CategoryMapper"
    case_name = "test_cc_category_mapper_benchmark"
    case_mode = "BENCHMARK"
    backends = ("onnxruntime", "onnx-light")
