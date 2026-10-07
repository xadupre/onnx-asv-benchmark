from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GroupqueryattentionBfloat16Bfloat16Bfloat16Int32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "group"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'int32', 'int32')


class GroupqueryattentionFloat16Float16Float16Int32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "group"
    case_dtypes = ('float16', 'float16', 'float16', 'int32', 'int32')


class GroupqueryattentionFloat32Float32Float32Float32Float32Int32Int32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "group"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32', 'int32', 'int32', 'float32', 'float32')


class GroupqueryattentionFloat32Float32Float32Int32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "group"
    case_dtypes = ('float32', 'float32', 'float32', 'int32', 'int32')
