from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LinearAttentionBfloat16Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "linear"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class LinearAttentionBfloat16Inputs6(_CpuBackendCaseBenchmark):
    case_prefix = "linear"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class LinearAttentionFloat32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "linear"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32')


class LinearAttentionFloat32Inputs6(_CpuBackendCaseBenchmark):
    case_prefix = "linear"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32', 'float32')


class LinearAttentionFloat32Inputs6Microsoft(_CpuBackendCaseBenchmark):
    case_prefix = "microsoft"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32', 'float32')
