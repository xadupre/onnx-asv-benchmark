from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GreaterPart01(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_start = 0
    case_stop = 200


class GreaterPart02(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_start = 200
    case_stop = 400


class GreaterPart03(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_start = 400
    case_stop = 600


class GreaterPart04(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_start = 600
    case_stop = None
