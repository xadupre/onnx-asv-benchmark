from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Exp(_CpuBackendCaseBenchmark):
    case_prefix = "exp"
    case_start = 0
    case_stop = None
