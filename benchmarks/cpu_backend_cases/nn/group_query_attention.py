from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GroupQueryAttention(_CpuBackendCaseBenchmark):
    case_prefix = "group"
    case_start = 0
    case_stop = None
