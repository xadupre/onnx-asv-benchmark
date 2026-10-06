from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MatMulNBits(_CpuBackendCaseBenchmark):
    case_prefix = "matmulnbits"
    case_start = 0
    case_stop = None
