from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class PreluBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('bfloat16', 'bfloat16')


class PreluFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('float16', 'float16')


class PreluFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('float32', 'float32')


class PreluFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('float64', 'float64')


class PreluInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('int32', 'int32')


class PreluInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('int64', 'int64')


class PreluUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('uint32', 'uint32')


class PreluUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "prelu"
    case_dtypes = ('uint64', 'uint64')
