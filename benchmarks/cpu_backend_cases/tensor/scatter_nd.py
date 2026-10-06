from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class ScatterND(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_start = 0
    case_stop = None
