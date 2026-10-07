from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LinearattentionBfloat16Bfloat16Bfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "linear"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class LinearattentionBfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "linear"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class LinearattentionFloat32Float32Float32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "linear"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32')


class LinearattentionFloat32Float32Float32Float32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "linear"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32', 'float32')


class LinearattentionFloat32Float32Float32Float32Float32Float32Microsoft(_CpuBackendCaseBenchmark):
    case_prefix = "microsoft"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32', 'float32')
