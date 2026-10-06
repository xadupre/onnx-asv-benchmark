from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class AddPart01(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_start = 0
    case_stop = 200


class AddPart02(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_start = 200
    case_stop = None
