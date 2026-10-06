from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LessOrEqualPart01(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_start = 0
    case_stop = 200


class LessOrEqualPart02(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_start = 200
    case_stop = 400


class LessOrEqualPart03(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_start = 400
    case_stop = 600


class LessOrEqualPart04(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_start = 600
    case_stop = None
