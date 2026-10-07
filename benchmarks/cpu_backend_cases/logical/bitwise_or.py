from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitwiseOrInt16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('int16', 'int16')


class BitwiseOrInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('int32', 'int32')


class BitwiseOrInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('int64', 'int64')


class BitwiseOrInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('int8', 'int8')


class BitwiseOrUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('uint16', 'uint16')


class BitwiseOrUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('uint32', 'uint32')


class BitwiseOrUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('uint64', 'uint64')


class BitwiseOrUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwiseor"
    case_dtypes = ('uint8', 'uint8')
