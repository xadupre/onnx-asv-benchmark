from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MatmulintegerInt8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "matmulinteger"
    case_dtypes = ('int8', 'int8')


class MatmulintegerInt8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "matmulinteger"
    case_dtypes = ('int8', 'uint8')


class MatmulintegerUint8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "matmulinteger"
    case_dtypes = ('uint8', 'int8')


class MatmulintegerUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "matmulinteger"
    case_dtypes = ('uint8', 'uint8')
