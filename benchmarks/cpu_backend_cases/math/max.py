from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MaxFloat32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "max"
    case_dtypes = ('float32', 'float32', 'float32')
