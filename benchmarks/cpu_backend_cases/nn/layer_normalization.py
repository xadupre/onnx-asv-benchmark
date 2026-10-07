from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LayerNormalizationBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "layernormalization"
    case_dtypes = ('bfloat16', 'bfloat16')


class LayerNormalizationFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "layernormalization"
    case_dtypes = ('float16', 'float16')


class LayerNormalizationFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "layernormalization"
    case_dtypes = ('float32', 'float32')


class LayerNormalizationFloat32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "layernormalization"
    case_dtypes = ('float32', 'float32', 'float32')
