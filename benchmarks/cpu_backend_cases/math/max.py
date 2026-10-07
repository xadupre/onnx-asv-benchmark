from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Max(_CpuBackendCaseBenchmark):
    case_prefix = "max"
    case_names = (
        'test_cpu_max_n4096_3inputs_float32_benchmark',
    )
