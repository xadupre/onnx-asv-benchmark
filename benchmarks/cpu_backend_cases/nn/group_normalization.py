from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GroupNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "groupnormalization"
    case_start = 0
    case_stop = None
