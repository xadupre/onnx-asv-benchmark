from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GatherPart01(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_start = 0
    case_stop = 100


class GatherPart02(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_start = 100
    case_stop = None
