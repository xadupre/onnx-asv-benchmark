from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GreaterOrEqualBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_dtypes = ('bfloat16', 'bfloat16')


class GreaterOrEqualFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_dtypes = ('float16', 'float16')


class GreaterOrEqualFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_dtypes = ('float32', 'float32')


class GreaterOrEqualInt16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_dtypes = ('int16', 'int16')


class GreaterOrEqualInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_dtypes = ('int32', 'int32')


class GreaterOrEqualInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_dtypes = ('int64', 'int64')


class GreaterOrEqualInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_dtypes = ('int8', 'int8')


class GreaterOrEqualUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_dtypes = ('uint16', 'uint16')


class GreaterOrEqualUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_dtypes = ('uint32', 'uint32')


class GreaterOrEqualUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_dtypes = ('uint64', 'uint64')


class GreaterOrEqualUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "greaterorequal"
    case_dtypes = ('uint8', 'uint8')
