from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class CDistFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "cdist"
    case_dtypes = ('float32', 'float32')


class CDistFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "cdist"
    case_dtypes = ('float64', 'float64')
