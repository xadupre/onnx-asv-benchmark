from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BiasGelu(_CpuBackendCaseBenchmark):
    case_prefix = "biasgelu"
    case_start = 0
    case_stop = None
