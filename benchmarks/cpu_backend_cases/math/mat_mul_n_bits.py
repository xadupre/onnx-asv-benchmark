from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MatMulNBitsBfloat16Uint8Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "matmulnbits"
    case_dtypes = ('bfloat16', 'uint8', 'bfloat16')


class MatMulNBitsFloat16Uint8Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "matmulnbits"
    case_dtypes = ('float16', 'uint8', 'float16')


class MatMulNBitsFloat32Uint8Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "matmulnbits"
    case_dtypes = ('float32', 'uint8', 'float32')
