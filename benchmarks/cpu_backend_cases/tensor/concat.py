from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class ConcatBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('bfloat16',)


class ConcatBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('bfloat16', 'bfloat16')


class ConcatBfloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16')


class ConcatBfloat16Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class ConcatBfloat16Inputs32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class ConcatFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float16',)


class ConcatFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float16', 'float16')


class ConcatFloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float16', 'float16', 'float16')


class ConcatFloat16Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float16', 'float16', 'float16', 'float16')


class ConcatFloat16Inputs32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16')


class ConcatFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float32',)


class ConcatFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float32', 'float32')


class ConcatFloat32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float32', 'float32', 'float32')


class ConcatFloat32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float32', 'float32', 'float32', 'float32')


class ConcatFloat32Inputs32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32')


class ConcatFloat64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float64',)


class ConcatFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float64', 'float64')


class ConcatFloat64Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float64', 'float64', 'float64')


class ConcatFloat64Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float64', 'float64', 'float64', 'float64')


class ConcatFloat64Inputs32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64')


class ConcatInt64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int64',)


class ConcatInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int64', 'int64')


class ConcatInt64Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int64', 'int64', 'int64')


class ConcatInt64Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int64', 'int64', 'int64', 'int64')


class ConcatInt64Inputs32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64')


class ConcatInt8(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int8',)


class ConcatInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int8', 'int8')


class ConcatInt8Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int8', 'int8', 'int8')


class ConcatInt8Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int8', 'int8', 'int8', 'int8')


class ConcatInt8Inputs32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8')
