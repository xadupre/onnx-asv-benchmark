from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class CastBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_dtypes = ('bfloat16',)


class CastBool(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_dtypes = ('bool',)


class CastFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_dtypes = ('float16',)


class CastFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_dtypes = ('float32',)


class CastFloat64(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_dtypes = ('float64',)


class CastInt16(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_dtypes = ('int16',)


class CastInt32(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_dtypes = ('int32',)


class CastInt64(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_dtypes = ('int64',)


class CastInt8(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_dtypes = ('int8',)


class CastUint8(_CpuBackendCaseBenchmark):
    case_prefix = "cast"
    case_dtypes = ('uint8',)
