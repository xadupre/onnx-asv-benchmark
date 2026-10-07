from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class ModPart01(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_start = 0
    case_stop = 100


class ModPart02(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_start = 100
    case_stop = 200


class ModPart03(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_start = 200
    case_stop = 300


class ModPart04(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_start = 300
    case_stop = None
