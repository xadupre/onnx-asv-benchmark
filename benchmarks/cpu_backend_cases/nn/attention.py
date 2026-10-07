from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class AttentionBfloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16')


class AttentionBfloat16Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class AttentionBfloat16BoolInputs4(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bool')


class AttentionBfloat16Float32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'float32')


class AttentionBfloat16Int64Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'int64')


class AttentionFloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float16', 'float16', 'float16')


class AttentionFloat16BoolInputs4(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float16', 'float16', 'float16', 'bool')


class AttentionFloat16Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float16', 'float16', 'float16', 'float16', 'float16')


class AttentionFloat16Float32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float16', 'float16', 'float16', 'float32')


class AttentionFloat16Int64Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float16', 'float16', 'float16', 'int64')


class AttentionFloat32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float32', 'float32', 'float32')


class AttentionFloat32BoolInputs4(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float32', 'float32', 'float32', 'bool')


class AttentionFloat32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float32', 'float32', 'float32', 'float32')


class AttentionFloat32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32')


class AttentionFloat32Int64Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "attention"
    case_dtypes = ('float32', 'float32', 'float32', 'int64')
