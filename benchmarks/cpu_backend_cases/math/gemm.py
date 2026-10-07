from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GemmBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('bfloat16', 'bfloat16')


class GemmBfloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16')


class GemmFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float16', 'float16')


class GemmFloat16Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float16', 'float16', 'float16')


class GemmFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float32', 'float32')


class GemmFloat32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float32', 'float32', 'float32')


class GemmFloat32Inputs4(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float32', 'float32', 'float32', 'float32')


class GemmFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float64', 'float64')


class GemmFloat64Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float64', 'float64', 'float64')
