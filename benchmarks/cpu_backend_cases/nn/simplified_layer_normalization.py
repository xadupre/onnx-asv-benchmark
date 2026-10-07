from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SimplifiedlayernormalizationBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "simplified"
    case_dtypes = ('bfloat16', 'bfloat16')


class SimplifiedlayernormalizationFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "simplified"
    case_dtypes = ('float16', 'float16')


class SimplifiedlayernormalizationFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "simplified"
    case_dtypes = ('float32', 'float32')


class SimplifiedlayernormalizationFloat64Float64(_CpuBackendCaseBenchmark):
    case_prefix = "simplified"
    case_dtypes = ('float64', 'float64')
