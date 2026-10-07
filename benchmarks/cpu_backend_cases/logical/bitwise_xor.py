from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitwiseXorInt16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('int16', 'int16')


class BitwiseXorInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('int32', 'int32')


class BitwiseXorInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('int64', 'int64')


class BitwiseXorInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('int8', 'int8')


class BitwiseXorUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('uint16', 'uint16')


class BitwiseXorUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('uint32', 'uint32')


class BitwiseXorUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('uint64', 'uint64')


class BitwiseXorUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('uint8', 'uint8')
