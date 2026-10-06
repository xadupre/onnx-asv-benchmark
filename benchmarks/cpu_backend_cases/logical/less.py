from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LessPart01(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_start = 0
    case_stop = 200


class LessPart02(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_start = 200
    case_stop = 400


class LessPart03(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_start = 400
    case_stop = 600


class LessPart04(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_start = 600
    case_stop = None
