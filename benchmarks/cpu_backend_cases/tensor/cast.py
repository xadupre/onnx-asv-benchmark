from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class CastPart01(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_start = 0
    case_stop = 100


class CastPart02(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_start = 100
    case_stop = None
