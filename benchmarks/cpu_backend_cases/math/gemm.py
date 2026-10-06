from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Gemm(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_start = 0
    case_stop = None
