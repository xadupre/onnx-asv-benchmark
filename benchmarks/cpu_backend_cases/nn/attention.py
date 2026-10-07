from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class AttentionBfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16')


class AttentionBfloat16Bfloat16Bfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class AttentionBfloat16Bfloat16Bfloat16Bool(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bool')


class AttentionBfloat16Bfloat16Bfloat16Float32(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'float32')


class AttentionBfloat16Bfloat16Bfloat16Int64(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'int64')


class AttentionFloat16Float16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float16', 'float16', 'float16')


class AttentionFloat16Float16Float16Bool(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float16', 'float16', 'float16', 'bool')


class AttentionFloat16Float16Float16Float16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float16', 'float16', 'float16', 'float16', 'float16')


class AttentionFloat16Float16Float16Float32(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float16', 'float16', 'float16', 'float32')


class AttentionFloat16Float16Float16Int64(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float16', 'float16', 'float16', 'int64')


class AttentionFloat32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float32', 'float32', 'float32')


class AttentionFloat32Float32Float32Bool(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float32', 'float32', 'float32', 'bool')


class AttentionFloat32Float32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float32', 'float32', 'float32', 'float32')


class AttentionFloat32Float32Float32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32')


class AttentionFloat32Float32Float32Int64(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float32', 'float32', 'float32', 'int64')
