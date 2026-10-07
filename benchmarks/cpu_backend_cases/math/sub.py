from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SubBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('bfloat16', 'bfloat16')


class SubFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('float16', 'float16')


class SubFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('float32', 'float32')


class SubFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('float64', 'float64')


class SubInt16Int16(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('int16', 'int16')


class SubInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('int32', 'int32')


class SubInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('int64', 'int64')


class SubInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('int8', 'int8')


class SubUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('uint16', 'uint16')


class SubUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('uint32', 'uint32')


class SubUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('uint64', 'uint64')


class SubUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('uint8', 'uint8')
