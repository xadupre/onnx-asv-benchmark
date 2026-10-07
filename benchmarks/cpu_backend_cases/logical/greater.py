from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GreaterBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_dtypes = ('bfloat16', 'bfloat16')


class GreaterFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_dtypes = ('float16', 'float16')


class GreaterFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_dtypes = ('float32', 'float32')


class GreaterInt16Int16(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_dtypes = ('int16', 'int16')


class GreaterInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_dtypes = ('int32', 'int32')


class GreaterInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_dtypes = ('int64', 'int64')


class GreaterInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_dtypes = ('int8', 'int8')


class GreaterUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_dtypes = ('uint16', 'uint16')


class GreaterUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_dtypes = ('uint32', 'uint32')


class GreaterUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_dtypes = ('uint64', 'uint64')


class GreaterUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "greater"
    case_dtypes = ('uint8', 'uint8')
