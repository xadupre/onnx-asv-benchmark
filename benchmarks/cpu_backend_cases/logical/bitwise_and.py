from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitwiseAndPart01(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_start = 0
    case_stop = 200


class BitwiseAndPart02(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_start = 200
    case_stop = None
