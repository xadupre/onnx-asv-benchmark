from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Min(_CpuBackendCaseBenchmark):
    case_prefix = "min"
    case_names = (
        'test_cpu_min_n4096_3inputs_float32_benchmark',
    )
