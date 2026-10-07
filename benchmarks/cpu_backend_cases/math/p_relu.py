from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class PReluPart01(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_start = 0
    case_stop = 100


class PReluPart02(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_start = 100
    case_stop = 200


class PReluPart03(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_start = 200
    case_stop = None
