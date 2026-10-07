from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class EqualPart01(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_start = 0
    case_stop = 100


class EqualPart02(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_start = 100
    case_stop = 200


class EqualPart03(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_start = 200
    case_stop = 300


class EqualPart04(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_start = 300
    case_stop = None
