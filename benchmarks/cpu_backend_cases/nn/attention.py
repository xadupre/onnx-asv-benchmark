from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Attention(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_start = 0
    case_stop = None
