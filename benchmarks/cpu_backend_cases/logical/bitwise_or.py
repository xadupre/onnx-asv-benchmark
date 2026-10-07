from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitwiseorInt16Int16(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('int16', 'int16')


class BitwiseorInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('int32', 'int32')


class BitwiseorInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('int64', 'int64')


class BitwiseorInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('int8', 'int8')


class BitwiseorUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('uint16', 'uint16')


class BitwiseorUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('uint32', 'uint32')


class BitwiseorUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('uint64', 'uint64')


class BitwiseorUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('uint8', 'uint8')
