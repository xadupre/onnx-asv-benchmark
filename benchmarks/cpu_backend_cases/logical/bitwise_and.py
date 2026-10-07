from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitwiseAndInt16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('int16', 'int16')


class BitwiseAndInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('int32', 'int32')


class BitwiseAndInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('int64', 'int64')


class BitwiseAndInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('int8', 'int8')


class BitwiseAndUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('uint16', 'uint16')


class BitwiseAndUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('uint32', 'uint32')


class BitwiseAndUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('uint64', 'uint64')


class BitwiseAndUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseand"
    case_dtypes = ('uint8', 'uint8')
