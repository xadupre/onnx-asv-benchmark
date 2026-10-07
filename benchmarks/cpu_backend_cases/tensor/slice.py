from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SliceBfloat16Int32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('bfloat16', 'int32', 'int32', 'int32')


class SliceBfloat16Int32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('bfloat16', 'int32', 'int32', 'int32', 'int32')


class SliceBfloat16Int64Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('bfloat16', 'int64', 'int64', 'int64')


class SliceBfloat16Int64Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('bfloat16', 'int64', 'int64', 'int64', 'int64')


class SliceFloat16Int32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float16', 'int32', 'int32', 'int32')


class SliceFloat16Int32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float16', 'int32', 'int32', 'int32', 'int32')


class SliceFloat16Int64Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float16', 'int64', 'int64', 'int64')


class SliceFloat16Int64Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float16', 'int64', 'int64', 'int64', 'int64')


class SliceFloat32Int32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float32', 'int32', 'int32', 'int32')


class SliceFloat32Int32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float32', 'int32', 'int32', 'int32', 'int32')


class SliceFloat32Int64Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float32', 'int64', 'int64', 'int64')


class SliceFloat32Int64Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float32', 'int64', 'int64', 'int64', 'int64')


class SliceFloat64Int32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float64', 'int32', 'int32', 'int32')


class SliceFloat64Int32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float64', 'int32', 'int32', 'int32', 'int32')


class SliceFloat64Int64Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float64', 'int64', 'int64', 'int64')


class SliceFloat64Int64Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('float64', 'int64', 'int64', 'int64', 'int64')


class SliceInt64Int32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('int64', 'int32', 'int32', 'int32')


class SliceInt64Int32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('int64', 'int32', 'int32', 'int32', 'int32')


class SliceInt64Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('int64', 'int64', 'int64', 'int64')


class SliceInt64Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('int64', 'int64', 'int64', 'int64', 'int64')


class SliceInt8Int32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('int8', 'int32', 'int32', 'int32')


class SliceInt8Int32Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('int8', 'int32', 'int32', 'int32', 'int32')


class SliceInt8Int64Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('int8', 'int64', 'int64', 'int64')


class SliceInt8Int64Inputs5(_CpuBackendCaseBenchmark):
    case_prefix = "slice"
    case_dtypes = ('int8', 'int64', 'int64', 'int64', 'int64')
