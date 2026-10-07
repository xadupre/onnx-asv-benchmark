from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class TreeEnsembleFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "treeensemble"
    case_dtypes = ('float32',)
