from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Gather(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_start = 0
    case_stop = None
