from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GemmBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('bfloat16', 'bfloat16')


class GemmBfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16')


class GemmFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float16', 'float16')


class GemmFloat16Float16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float16', 'float16', 'float16')


class GemmFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float32', 'float32')


class GemmFloat32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float32', 'float32', 'float32')


class GemmFloat32Float32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float32', 'float32', 'float32', 'float32')


class GemmFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float64', 'float64')


class GemmFloat64Float64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "gemm"
    case_dtypes = ('float64', 'float64', 'float64')
