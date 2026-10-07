from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GatherBfloat16Int32(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('bfloat16', 'int32')


class GatherBfloat16Int64(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('bfloat16', 'int64')


class GatherFloat16Int32(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('float16', 'int32')


class GatherFloat16Int64(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('float16', 'int64')


class GatherFloat32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('float32', 'int32')


class GatherFloat32Int64(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('float32', 'int64')


class GatherFloat64Int32(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('float64', 'int32')


class GatherFloat64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('float64', 'int64')


class GatherInt64Int32(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('int64', 'int32')


class GatherInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('int64', 'int64')


class GatherInt8Int32(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('int8', 'int32')


class GatherInt8Int64(_CpuBackendCaseBenchmark):
    case_prefix = "gather"
    case_dtypes = ('int8', 'int64')
