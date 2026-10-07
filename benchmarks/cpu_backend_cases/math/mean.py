from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Mean(_CpuBackendCaseBenchmark):
    case_prefix = "mean"
    case_names = (
        'test_cpu_mean_n4096_3inputs_float32_benchmark',
        'test_cpu_mean_n4096_3inputs_float64_benchmark',
    )
