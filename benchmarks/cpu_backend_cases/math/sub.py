from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SubPart01(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_start = 0
    case_stop = 200


class SubPart02(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_start = 200
    case_stop = 400


class SubPart03(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_start = 400
    case_stop = 600


class SubPart04(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_start = 600
    case_stop = None
