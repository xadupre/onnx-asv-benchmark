from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MatmulBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "matmul"
    case_dtypes = ('bfloat16', 'bfloat16')


class MatmulFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "matmul"
    case_dtypes = ('float16', 'float16')


class MatmulFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "matmul"
    case_dtypes = ('float32', 'float32')


class MatmulFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "matmul"
    case_dtypes = ('float64', 'float64')
