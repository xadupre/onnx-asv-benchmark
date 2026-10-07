from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitshiftUint16Uint16(_CpuBackendCaseBenchmark):
    case_prefix = "bitshift"
    case_dtypes = ('uint16', 'uint16')


class BitshiftUint32Uint32(_CpuBackendCaseBenchmark):
    case_prefix = "bitshift"
    case_dtypes = ('uint32', 'uint32')


class BitshiftUint64Uint64(_CpuBackendCaseBenchmark):
    case_prefix = "bitshift"
    case_dtypes = ('uint64', 'uint64')


class BitshiftUint8Uint8(_CpuBackendCaseBenchmark):
    case_prefix = "bitshift"
    case_dtypes = ('uint8', 'uint8')
