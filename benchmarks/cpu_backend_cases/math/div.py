from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class DivBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('bfloat16', 'bfloat16')


class DivFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('float16', 'float16')


class DivFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('float32', 'float32')


class DivFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('float64', 'float64')


class DivInt16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('int16', 'int16')


class DivInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('int32', 'int32')


class DivInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('int64', 'int64')


class DivInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('int8', 'int8')


class DivUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('uint16', 'uint16')


class DivUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('uint32', 'uint32')


class DivUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('uint64', 'uint64')


class DivUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "div"
    case_dtypes = ('uint8', 'uint8')
