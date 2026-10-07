from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LessBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('bfloat16', 'bfloat16')


class LessFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('float16', 'float16')


class LessFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('float32', 'float32')


class LessInt16Int16(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('int16', 'int16')


class LessInt32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('int32', 'int32')


class LessInt64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('int64', 'int64')


class LessInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('int8', 'int8')


class LessUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('uint16', 'uint16')


class LessUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('uint32', 'uint32')


class LessUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('uint64', 'uint64')


class LessUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "less"
    case_dtypes = ('uint8', 'uint8')
