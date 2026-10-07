from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class EqualBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('bfloat16', 'bfloat16')


class EqualBoolBool(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('bool', 'bool')


class EqualFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('float16', 'float16')


class EqualFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('float32', 'float32')


class EqualFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('float64', 'float64')


class EqualInt16Int16(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('int16', 'int16')


class EqualInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('int32', 'int32')


class EqualInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('int64', 'int64')


class EqualInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('int8', 'int8')


class EqualUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('uint16', 'uint16')


class EqualUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('uint32', 'uint32')


class EqualUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('uint64', 'uint64')


class EqualUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('uint8', 'uint8')
