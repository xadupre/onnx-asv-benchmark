from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MulBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('bfloat16', 'bfloat16')


class MulFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('float16', 'float16')


class MulFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('float32', 'float32')


class MulFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('float64', 'float64')


class MulInt16Int16(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('int16', 'int16')


class MulInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('int32', 'int32')


class MulInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('int64', 'int64')


class MulInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('int8', 'int8')


class MulUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('uint16', 'uint16')


class MulUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('uint32', 'uint32')


class MulUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('uint64', 'uint64')


class MulUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('uint8', 'uint8')
