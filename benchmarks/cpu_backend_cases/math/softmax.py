from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Softmax(_CpuBackendCaseBenchmark):
    case_prefix = "softmax"
    case_start = 0
    case_stop = None
