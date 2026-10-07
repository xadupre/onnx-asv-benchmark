from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SlicePart01(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_start = 0
    case_stop = 100


class SlicePart02(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_start = 100
    case_stop = None
