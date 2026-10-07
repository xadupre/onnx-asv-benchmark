from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SoftmaxBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "softmax"
    case_dtypes = ('bfloat16',)


class SoftmaxFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "softmax"
    case_dtypes = ('float16',)


class SoftmaxFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "softmax"
    case_dtypes = ('float32',)


class SoftmaxFloat64(_CpuBackendCaseBenchmark):
    case_prefix = "softmax"
    case_dtypes = ('float64',)
