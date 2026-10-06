from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class RMSNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "rms"
    case_start = 0
    case_stop = None
