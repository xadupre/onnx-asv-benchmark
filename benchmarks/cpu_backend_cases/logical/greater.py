from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GreaterPart01(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_start = 0
    case_stop = 100


class GreaterPart02(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_start = 100
    case_stop = 200


class GreaterPart03(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_start = 200
    case_stop = 300


class GreaterPart04(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_start = 300
    case_stop = 400


class GreaterPart05(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_start = 400
    case_stop = 500


class GreaterPart06(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_start = 500
    case_stop = 600


class GreaterPart07(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_start = 600
    case_stop = None
