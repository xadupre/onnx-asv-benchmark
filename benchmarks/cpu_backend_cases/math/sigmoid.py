from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SigmoidBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "sigmoid"
    case_dtypes = ('bfloat16',)


class SigmoidFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "sigmoid"
    case_dtypes = ('float16',)


class SigmoidFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "sigmoid"
    case_dtypes = ('float32',)


class SigmoidFloat64(_CpuBackendCaseBenchmark):
    case_prefix = "sigmoid"
    case_dtypes = ('float64',)
