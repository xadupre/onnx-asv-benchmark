from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GemmPart01(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_start = 0
    case_stop = 100


class GemmPart02(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_start = 100
    case_stop = None
