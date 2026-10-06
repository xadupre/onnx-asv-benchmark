from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Sigmoid(_CpuBackendCaseBenchmark):
    case_prefix = "sigmoid"
    case_start = 0
    case_stop = None
