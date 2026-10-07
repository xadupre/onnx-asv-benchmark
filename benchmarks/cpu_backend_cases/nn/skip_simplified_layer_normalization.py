from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SkipSimplifiedLayerNormalizationBfloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16')


class SkipSimplifiedLayerNormalizationBfloat16Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class SkipSimplifiedLayerNormalizationFloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('float16', 'float16', 'float16')


class SkipSimplifiedLayerNormalizationFloat16Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('float16', 'float16', 'float16', 'float16')


class SkipSimplifiedLayerNormalizationFloat32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('float32', 'float32', 'float32')


class SkipSimplifiedLayerNormalizationFloat32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('float32', 'float32', 'float32', 'float32')
