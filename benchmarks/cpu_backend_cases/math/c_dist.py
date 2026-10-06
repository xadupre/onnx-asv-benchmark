from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class CDist(_CpuBackendCaseBenchmark):
    case_prefix = "cdist"
    case_start = 0
    case_stop = None
