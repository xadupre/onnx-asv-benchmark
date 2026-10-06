from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LinearAttention(_CpuBackendCaseBenchmark):
    case_prefix = "linear"
    case_start = 0
    case_stop = None


class LinearAttentionMicrosoft(_CpuBackendCaseBenchmark):
    case_prefix = "microsoft"
    case_start = 0
    case_stop = None
