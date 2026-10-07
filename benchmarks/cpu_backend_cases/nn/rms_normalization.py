from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class RmsnormalizationBfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "rms"
    case_dtypes = ('bfloat16', 'bfloat16')


class RmsnormalizationFloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "rms"
    case_dtypes = ('float16', 'float16')


class RmsnormalizationFloat32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "rms"
    case_dtypes = ('float32', 'float32')
