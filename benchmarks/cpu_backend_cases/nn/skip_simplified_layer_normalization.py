from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SkipSimplifiedLayerNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_start = 0
    case_stop = None
