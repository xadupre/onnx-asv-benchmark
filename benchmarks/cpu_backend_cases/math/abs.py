from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Abs(_CpuBackendCaseBenchmark):
    case_prefix = "abs"
    case_start = 0
    case_stop = None
