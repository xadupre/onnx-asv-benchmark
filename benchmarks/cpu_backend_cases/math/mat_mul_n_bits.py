from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MatmulnbitsBfloat16Uint8Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "matmulnbits"
    case_dtypes = ('bfloat16', 'uint8', 'bfloat16')


class MatmulnbitsFloat16Uint8Float16(_CpuBackendCaseBenchmark):
    case_prefix = "matmulnbits"
    case_dtypes = ('float16', 'uint8', 'float16')


class MatmulnbitsFloat32Uint8Float32(_CpuBackendCaseBenchmark):
    case_prefix = "matmulnbits"
    case_dtypes = ('float32', 'uint8', 'float32')
