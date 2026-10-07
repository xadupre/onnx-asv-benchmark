from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class ConcatBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('bfloat16',)


class ConcatBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('bfloat16', 'bfloat16')


class ConcatBfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16')


class ConcatBfloat16Bfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class ConcatBfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class ConcatFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float16',)


class ConcatFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float16', 'float16')


class ConcatFloat16Float16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float16', 'float16', 'float16')


class ConcatFloat16Float16Float16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float16', 'float16', 'float16', 'float16')


class ConcatFloat16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16', 'float16')


class ConcatFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float32',)


class ConcatFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float32', 'float32')


class ConcatFloat32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float32', 'float32', 'float32')


class ConcatFloat32Float32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float32', 'float32', 'float32', 'float32')


class ConcatFloat32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32')


class ConcatFloat64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float64',)


class ConcatFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float64', 'float64')


class ConcatFloat64Float64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float64', 'float64', 'float64')


class ConcatFloat64Float64Float64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float64', 'float64', 'float64', 'float64')


class ConcatFloat64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64')


class ConcatInt64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int64',)


class ConcatInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int64', 'int64')


class ConcatInt64Int64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int64', 'int64', 'int64')


class ConcatInt64Int64Int64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int64', 'int64', 'int64', 'int64')


class ConcatInt64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64')


class ConcatInt8(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int8',)


class ConcatInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int8', 'int8')


class ConcatInt8Int8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int8', 'int8', 'int8')


class ConcatInt8Int8Int8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int8', 'int8', 'int8', 'int8')


class ConcatInt8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "concat"
    case_dtypes = ('int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8', 'int8')
