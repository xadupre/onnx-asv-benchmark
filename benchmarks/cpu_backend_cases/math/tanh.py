from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Tanh(_CpuBackendCaseBenchmark):
    case_prefix = "tanh"
    case_start = 0
    case_stop = None
