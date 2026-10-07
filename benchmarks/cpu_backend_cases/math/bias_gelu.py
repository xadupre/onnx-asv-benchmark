from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BiasGeluBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "biasgelu"
    case_dtypes = ('bfloat16', 'bfloat16')


class BiasGeluFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "biasgelu"
    case_dtypes = ('float16', 'float16')


class BiasGeluFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "biasgelu"
    case_dtypes = ('float32', 'float32')


class BiasGeluFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "biasgelu"
    case_dtypes = ('float64', 'float64')
