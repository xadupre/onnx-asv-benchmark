from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class TanhBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "tanh"
    case_dtypes = ('bfloat16',)


class TanhFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "tanh"
    case_dtypes = ('float16',)


class TanhFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "tanh"
    case_dtypes = ('float32',)
