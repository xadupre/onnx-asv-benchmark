from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Mean(_CpuBackendCaseBenchmark):
    case_prefix = "mean"
    case_start = 0
    case_stop = None
