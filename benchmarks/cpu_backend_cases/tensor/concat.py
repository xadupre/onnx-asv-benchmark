from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Concat(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_start = 0
    case_stop = None
