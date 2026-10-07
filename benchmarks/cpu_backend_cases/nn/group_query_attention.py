from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GroupQueryAttentionBfloat16Int32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "group"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'int32', 'int32')


class GroupQueryAttentionFloat16Int32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "group"
    case_dtypes = ('float16', 'float16', 'float16', 'int32', 'int32')


class GroupQueryAttentionFloat32Int32Inputs9(_CpuBackendCaseBenchmark):
    case_prefix = "group"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32', 'int32', 'int32', 'float32', 'float32')


class GroupQueryAttentionFloat32Int32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "group"
    case_dtypes = ('float32', 'float32', 'float32', 'int32', 'int32')
