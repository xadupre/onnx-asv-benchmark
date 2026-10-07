from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LessorequalBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('bfloat16', 'bfloat16')


class LessorequalFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('float16', 'float16')


class LessorequalFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('float32', 'float32')


class LessorequalInt16Int16(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('int16', 'int16')


class LessorequalInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('int32', 'int32')


class LessorequalInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('int64', 'int64')


class LessorequalInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('int8', 'int8')


class LessorequalUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('uint16', 'uint16')


class LessorequalUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('uint32', 'uint32')


class LessorequalUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('uint64', 'uint64')


class LessorequalUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('uint8', 'uint8')
