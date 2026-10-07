from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class ModBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('bfloat16', 'bfloat16')


class ModFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('float16', 'float16')


class ModFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('float32', 'float32')


class ModFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('float64', 'float64')


class ModInt16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('int16', 'int16')


class ModInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('int32', 'int32')


class ModInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('int64', 'int64')


class ModInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('int8', 'int8')


class ModUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('uint16', 'uint16')


class ModUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('uint32', 'uint32')


class ModUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('uint64', 'uint64')


class ModUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('uint8', 'uint8')
