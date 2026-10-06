from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MatMulInteger(_CpuBackendCaseBenchmark):
    case_prefix = "matmulinteger"
    case_start = 0
    case_stop = None
