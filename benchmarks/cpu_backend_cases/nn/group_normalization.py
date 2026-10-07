from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GroupNormalizationBfloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "groupnormalization"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16')


class GroupNormalizationFloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "groupnormalization"
    case_dtypes = ('float16', 'float16', 'float16')


class GroupNormalizationFloat32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "groupnormalization"
    case_dtypes = ('float32', 'float32', 'float32')
