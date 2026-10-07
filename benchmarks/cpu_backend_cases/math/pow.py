from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class PowBfloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('bfloat16', 'bfloat16')


class PowBfloat16Float16(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('bfloat16', 'float16')


class PowBfloat16Float32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('bfloat16', 'float32')


class PowBfloat16Int32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('bfloat16', 'int32')


class PowBfloat16Int64(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('bfloat16', 'int64')


class PowBfloat16Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('bfloat16', 'uint32')


class PowBfloat16Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('bfloat16', 'uint64')


class PowFloat16Bfloat16(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float16', 'bfloat16')


class PowFloat16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float16', 'float16')


class PowFloat16Float32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float16', 'float32')


class PowFloat16Int32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float16', 'int32')


class PowFloat16Int64(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float16', 'int64')


class PowFloat16Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float16', 'uint32')


class PowFloat16Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float16', 'uint64')


class PowFloat32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float32', 'float32')


class PowFloat32Int32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float32', 'int32')


class PowFloat32Int64(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float32', 'int64')


class PowFloat32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float32', 'uint32')


class PowFloat32Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('float32', 'uint64')


class PowInt32Float32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('int32', 'float32')


class PowInt32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('int32', 'int32')


class PowInt32Int64(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('int32', 'int64')


class PowInt32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('int32', 'uint32')


class PowInt32Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('int32', 'uint64')


class PowInt64Float32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('int64', 'float32')


class PowInt64Int32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('int64', 'int32')


class PowInt64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('int64', 'int64')


class PowInt64Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('int64', 'uint32')


class PowInt64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "pow"
    case_dtypes = ('int64', 'uint64')
