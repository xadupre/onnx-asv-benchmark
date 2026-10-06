from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Xor(_CpuBackendCaseBenchmark):
    case_prefix = "xor"
    case_start = 0
    case_stop = None
