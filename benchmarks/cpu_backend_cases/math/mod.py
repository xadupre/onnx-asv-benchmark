from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class ModBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('bfloat16', 'bfloat16')


class ModFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('float16', 'float16')


class ModFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('float32', 'float32')


class ModFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('float64', 'float64')


class ModInt16Int16(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('int16', 'int16')


class ModInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('int32', 'int32')


class ModInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('int64', 'int64')


class ModInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('int8', 'int8')


class ModUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('uint16', 'uint16')


class ModUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('uint32', 'uint32')


class ModUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('uint64', 'uint64')


class ModUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "mod"
    case_dtypes = ('uint8', 'uint8')
