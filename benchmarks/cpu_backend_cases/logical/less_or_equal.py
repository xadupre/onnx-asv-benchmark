from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LessOrEqualBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('bfloat16', 'bfloat16')


class LessOrEqualFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('float16', 'float16')


class LessOrEqualFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('float32', 'float32')


class LessOrEqualInt16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('int16', 'int16')


class LessOrEqualInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('int32', 'int32')


class LessOrEqualInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('int64', 'int64')


class LessOrEqualInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('int8', 'int8')


class LessOrEqualUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('uint16', 'uint16')


class LessOrEqualUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('uint32', 'uint32')


class LessOrEqualUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('uint64', 'uint64')


class LessOrEqualUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "lessorequal"
    case_dtypes = ('uint8', 'uint8')
