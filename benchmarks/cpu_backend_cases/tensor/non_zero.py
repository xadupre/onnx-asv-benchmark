from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class NonZero(_CpuBackendCaseBenchmark):
    case_prefix = "nonzero"
    case_start = 0
    case_stop = None
