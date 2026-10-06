from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Slice(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_start = 0
    case_stop = None
