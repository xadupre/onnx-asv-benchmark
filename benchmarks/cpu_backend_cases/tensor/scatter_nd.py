from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class ScatterndBfloat16Int32Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('bfloat16', 'int32', 'bfloat16')


class ScatterndBfloat16Int64Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('bfloat16', 'int64', 'bfloat16')


class ScatterndFloat16Int32Float16(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('float16', 'int32', 'float16')


class ScatterndFloat16Int64Float16(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('float16', 'int64', 'float16')


class ScatterndFloat32Int32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('float32', 'int32', 'float32')


class ScatterndFloat32Int64Float32(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('float32', 'int64', 'float32')
