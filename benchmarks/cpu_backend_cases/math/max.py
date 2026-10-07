from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MaxFloat32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "max"
    case_dtypes = ('float32', 'float32', 'float32')
