from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Softmax(_CpuBackendCaseBenchmark):
    case_prefix = "softmax"
    case_names = (
        'test_cpu_softmax_1024x1024_bfloat16_benchmark',
        'test_cpu_softmax_1024x1024_float16_benchmark',
        'test_cpu_softmax_1024x1024_float32_benchmark',
        'test_cpu_softmax_1024x1024_float64_benchmark',
        'test_cpu_softmax_1x1024_bfloat16_benchmark',
        'test_cpu_softmax_1x1024_float16_benchmark',
        'test_cpu_softmax_1x1024_float32_benchmark',
        'test_cpu_softmax_1x1024_float64_benchmark',
        'test_cpu_softmax_32x1024_bfloat16_benchmark',
        'test_cpu_softmax_32x1024_float16_benchmark',
        'test_cpu_softmax_32x1024_float32_benchmark',
        'test_cpu_softmax_32x1024_float64_benchmark',
    )
