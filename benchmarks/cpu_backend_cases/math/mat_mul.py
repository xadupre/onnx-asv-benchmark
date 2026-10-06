from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MatMul(_CpuBackendCaseBenchmark):
    case_prefix = "matmul"
    case_start = 0
    case_stop = None
