from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SkipsimplifiedlayernormalizationBfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16')


class SkipsimplifiedlayernormalizationBfloat16Bfloat16Bfloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')


class SkipsimplifiedlayernormalizationFloat16Float16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('float16', 'float16', 'float16')


class SkipsimplifiedlayernormalizationFloat16Float16Float16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('float16', 'float16', 'float16', 'float16')


class SkipsimplifiedlayernormalizationFloat32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('float32', 'float32', 'float32')


class SkipsimplifiedlayernormalizationFloat32Float32Float32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_dtypes = ('float32', 'float32', 'float32', 'float32')
