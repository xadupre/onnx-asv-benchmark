from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitwisexorInt16Int16(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('int16', 'int16')


class BitwisexorInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('int32', 'int32')


class BitwisexorInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('int64', 'int64')


class BitwisexorInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('int8', 'int8')


class BitwisexorUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('uint16', 'uint16')


class BitwisexorUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('uint32', 'uint32')


class BitwisexorUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('uint64', 'uint64')


class BitwisexorUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "bitwisexor"
    case_dtypes = ('uint8', 'uint8')
