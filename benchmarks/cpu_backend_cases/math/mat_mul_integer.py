from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MatMulIntegerInt8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "matmulinteger"
    case_dtypes = ('int8', 'int8')


class MatMulIntegerInt8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "matmulinteger"
    case_dtypes = ('int8', 'uint8')


class MatMulIntegerUint8Int8(_CpuBackendCaseBenchmark):
    case_prefix = "matmulinteger"
    case_dtypes = ('uint8', 'int8')


class MatMulIntegerUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "matmulinteger"
    case_dtypes = ('uint8', 'uint8')
