from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BiasGelu(_CpuBackendCaseBenchmark):
    case_prefix = "biasgelu"
    case_names = (
        'test_cpu_biasgelu_o1024_i1024_bfloat16_benchmark',
        'test_cpu_biasgelu_o1024_i1024_float16_benchmark',
        'test_cpu_biasgelu_o1024_i1024_float32_benchmark',
        'test_cpu_biasgelu_o1024_i1024_float64_benchmark',
        'test_cpu_biasgelu_o256_i4096_bfloat16_benchmark',
        'test_cpu_biasgelu_o256_i4096_float16_benchmark',
        'test_cpu_biasgelu_o256_i4096_float32_benchmark',
        'test_cpu_biasgelu_o256_i4096_float64_benchmark',
        'test_cpu_biasgelu_o4096_i256_bfloat16_benchmark',
        'test_cpu_biasgelu_o4096_i256_float16_benchmark',
        'test_cpu_biasgelu_o4096_i256_float32_benchmark',
        'test_cpu_biasgelu_o4096_i256_float64_benchmark',
    )
