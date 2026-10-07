from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class ScatterNDBfloat16Int32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('bfloat16', 'int32', 'bfloat16')


class ScatterNDBfloat16Int64Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('bfloat16', 'int64', 'bfloat16')


class ScatterNDFloat16Int32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('float16', 'int32', 'float16')


class ScatterNDFloat16Int64Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('float16', 'int64', 'float16')


class ScatterNDFloat32Int32Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('float32', 'int32', 'float32')


class ScatterNDFloat32Int64Inputs3(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_dtypes = ('float32', 'int64', 'float32')
