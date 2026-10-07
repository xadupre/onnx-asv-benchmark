from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SwigluBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "swiglu"
    case_dtypes = ('bfloat16', 'bfloat16')


class SwigluFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "swiglu"
    case_dtypes = ('float16', 'float16')


class SwigluFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "swiglu"
    case_dtypes = ('float32', 'float32')


class SwigluFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "swiglu"
    case_dtypes = ('float64', 'float64')
