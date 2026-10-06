from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitwiseOrPart01(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_start = 0
    case_stop = 200


class BitwiseOrPart02(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_start = 200
    case_stop = None
