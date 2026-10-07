from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SumFloat32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "sum"
    case_dtypes = ('float32', 'float32', 'float32')


class SumFloat64Float64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "sum"
    case_dtypes = ('float64', 'float64', 'float64')
