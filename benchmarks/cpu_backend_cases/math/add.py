from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class AddPart01(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_start = 0
    case_stop = 100


class AddPart02(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_start = 100
    case_stop = 200


class AddPart03(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_start = 200
    case_stop = 300


class AddPart04(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_start = 300
    case_stop = None
