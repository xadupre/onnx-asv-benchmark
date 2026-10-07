from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SimplifiedLayerNormalizationBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "simplified"
    case_dtypes = ('bfloat16', 'bfloat16')


class SimplifiedLayerNormalizationFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "simplified"
    case_dtypes = ('float16', 'float16')


class SimplifiedLayerNormalizationFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "simplified"
    case_dtypes = ('float32', 'float32')


class SimplifiedLayerNormalizationFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "simplified"
    case_dtypes = ('float64', 'float64')
