from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SubBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('bfloat16', 'bfloat16')


class SubFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('float16', 'float16')


class SubFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('float32', 'float32')


class SubFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('float64', 'float64')


class SubInt16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('int16', 'int16')


class SubInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('int32', 'int32')


class SubInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('int64', 'int64')


class SubInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('int8', 'int8')


class SubUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('uint16', 'uint16')


class SubUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('uint32', 'uint32')


class SubUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('uint64', 'uint64')


class SubUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "sub"
    case_dtypes = ('uint8', 'uint8')
