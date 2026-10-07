from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class CdistFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "cdist"
    case_dtypes = ('float32', 'float32')


class CdistFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "cdist"
    case_dtypes = ('float64', 'float64')
