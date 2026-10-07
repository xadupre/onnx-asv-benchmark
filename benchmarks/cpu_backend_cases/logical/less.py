from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LessPart01(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_start = 0
    case_stop = 100


class LessPart02(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_start = 100
    case_stop = 200


class LessPart03(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_start = 200
    case_stop = 300


class LessPart04(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_start = 300
    case_stop = 400


class LessPart05(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_start = 400
    case_stop = 500


class LessPart06(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_start = 500
    case_stop = 600


class LessPart07(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_start = 600
    case_stop = None
