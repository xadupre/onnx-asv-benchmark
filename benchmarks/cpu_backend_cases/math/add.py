from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class AddBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('bfloat16', 'bfloat16')


class AddFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('float16', 'float16')


class AddFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('float32', 'float32')


class AddFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('float64', 'float64')


class AddInt16Int16(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('int16', 'int16')


class AddInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('int32', 'int32')


class AddInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('int64', 'int64')


class AddInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('int8', 'int8')


class AddUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('uint16', 'uint16')


class AddUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('uint32', 'uint32')


class AddUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('uint64', 'uint64')


class AddUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "add"
    case_dtypes = ('uint8', 'uint8')
