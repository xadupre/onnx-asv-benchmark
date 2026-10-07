from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MatMulBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "matmul"
    case_dtypes = ('bfloat16', 'bfloat16')


class MatMulFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "matmul"
    case_dtypes = ('float16', 'float16')


class MatMulFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "matmul"
    case_dtypes = ('float32', 'float32')


class MatMulFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "matmul"
    case_dtypes = ('float64', 'float64')
