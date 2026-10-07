from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Sum(_CpuBackendCaseBenchmark):
    case_prefix = "sum"
    case_names = (
        'test_cpu_sum_n4096_3inputs_float32_benchmark',
        'test_cpu_sum_n4096_3inputs_float64_benchmark',
    )
