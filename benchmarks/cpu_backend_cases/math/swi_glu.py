from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SwiGLUBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "swiglu"
    case_dtypes = ('bfloat16', 'bfloat16')


class SwiGLUFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "swiglu"
    case_dtypes = ('float16', 'float16')


class SwiGLUFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "swiglu"
    case_dtypes = ('float32', 'float32')


class SwiGLUFloat64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "swiglu"
    case_dtypes = ('float64', 'float64')
