from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SplitBfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('bfloat16',)


class SplitBfloat16Int64(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('bfloat16', 'int64')


class SplitFloat16(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('float16',)


class SplitFloat16Int64(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('float16', 'int64')


class SplitFloat32(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('float32',)


class SplitFloat32Int64(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('float32', 'int64')


class SplitFloat64(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('float64',)


class SplitFloat64Int64(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('float64', 'int64')


class SplitInt64(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('int64',)


class SplitInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('int64', 'int64')


class SplitInt8(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('int8',)


class SplitInt8Int64(_CpuBackendCaseBenchmark):
    case_prefix = "split"
    case_dtypes = ('int8', 'int64')
