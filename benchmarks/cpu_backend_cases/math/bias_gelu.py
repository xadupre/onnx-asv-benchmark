from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BiasgeluBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "biasgelu"
    case_dtypes = ('bfloat16', 'bfloat16')


class BiasgeluFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "biasgelu"
    case_dtypes = ('float16', 'float16')


class BiasgeluFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "biasgelu"
    case_dtypes = ('float32', 'float32')


class BiasgeluFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "biasgelu"
    case_dtypes = ('float64', 'float64')
