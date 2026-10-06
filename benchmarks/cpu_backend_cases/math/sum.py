from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Sum(_CpuBackendCaseBenchmark):
    case_prefix = "sum"
    case_start = 0
    case_stop = None
