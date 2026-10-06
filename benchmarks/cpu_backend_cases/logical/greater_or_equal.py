from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GreaterOrEqualPart01(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_start = 0
    case_stop = 200


class GreaterOrEqualPart02(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_start = 200
    case_stop = 400


class GreaterOrEqualPart03(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_start = 400
    case_stop = 600


class GreaterOrEqualPart04(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_start = 600
    case_stop = None
