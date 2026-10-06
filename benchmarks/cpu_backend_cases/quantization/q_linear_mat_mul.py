from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class QLinearMatMul(_CpuBackendCaseBenchmark):
    case_prefix = "qlinearmatmul"
    case_start = 0
    case_stop = None
