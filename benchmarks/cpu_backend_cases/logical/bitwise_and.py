from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitwiseandInt16Int16(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('int16', 'int16')


class BitwiseandInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('int32', 'int32')


class BitwiseandInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('int64', 'int64')


class BitwiseandInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('int8', 'int8')


class BitwiseandUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('uint16', 'uint16')


class BitwiseandUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('uint32', 'uint32')


class BitwiseandUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('uint64', 'uint64')


class BitwiseandUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('uint8', 'uint8')
