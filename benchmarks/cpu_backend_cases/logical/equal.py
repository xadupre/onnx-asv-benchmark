from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class EqualBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('bfloat16', 'bfloat16')


class EqualBoolInputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('bool', 'bool')


class EqualFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('float16', 'float16')


class EqualFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('float32', 'float32')


class EqualFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('float64', 'float64')


class EqualInt16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('int16', 'int16')


class EqualInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('int32', 'int32')


class EqualInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('int64', 'int64')


class EqualInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('int8', 'int8')


class EqualUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('uint16', 'uint16')


class EqualUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('uint32', 'uint32')


class EqualUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('uint64', 'uint64')


class EqualUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "equal"
    case_dtypes = ('uint8', 'uint8')
