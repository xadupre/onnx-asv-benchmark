from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MeanVarianceNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "meanvariancenormalization"
    case_start = 0
    case_stop = None
