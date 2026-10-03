from benchmarks._operator import OperatorBenchmark as _OperatorBenchmark


class CategoryMapper(_OperatorBenchmark):
    operator = "CategoryMapper"
    case_name = "test_cc_category_mapper_benchmark"
    case_mode = "BENCHMARK"
    dtypes = ('int64',)
    backends = ("onnxruntime", "onnx-reference", "onnx-light")
