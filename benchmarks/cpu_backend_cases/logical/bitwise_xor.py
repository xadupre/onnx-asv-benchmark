from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitwiseXorPart01(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_start = 0
    case_stop = 200


class BitwiseXorPart02(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_start = 200
    case_stop = None
