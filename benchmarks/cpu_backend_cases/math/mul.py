from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MulBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('bfloat16', 'bfloat16')


class MulFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('float16', 'float16')


class MulFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('float32', 'float32')


class MulFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('float64', 'float64')


class MulInt16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('int16', 'int16')


class MulInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('int32', 'int32')


class MulInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('int64', 'int64')


class MulInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('int8', 'int8')


class MulUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('uint16', 'uint16')


class MulUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('uint32', 'uint32')


class MulUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('uint64', 'uint64')


class MulUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "mul"
    case_dtypes = ('uint8', 'uint8')
