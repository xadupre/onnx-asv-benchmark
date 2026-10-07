from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SumFloat32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "sum"
    case_dtypes = ('float32', 'float32', 'float32')


class SumFloat64Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "sum"
    case_dtypes = ('float64', 'float64', 'float64')
