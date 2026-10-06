from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class LayerNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "layernormalization"
    case_start = 0
    case_stop = None
