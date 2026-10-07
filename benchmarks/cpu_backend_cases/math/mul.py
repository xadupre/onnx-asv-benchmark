from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MulPart01(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_start = 0
    case_stop = 100


class MulPart02(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_start = 100
    case_stop = 200


class MulPart03(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_start = 200
    case_stop = 300


class MulPart04(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_start = 300
    case_stop = None
