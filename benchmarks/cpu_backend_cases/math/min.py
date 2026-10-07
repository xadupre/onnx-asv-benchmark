from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MinFloat32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "min"
    case_dtypes = ('float32', 'float32', 'float32')
