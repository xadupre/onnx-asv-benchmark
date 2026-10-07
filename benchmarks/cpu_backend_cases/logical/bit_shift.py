from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BitShiftUint16Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitshift"
    case_dtypes = ('uint16', 'uint16')


class BitShiftUint32Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitshift"
    case_dtypes = ('uint32', 'uint32')


class BitShiftUint64Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitshift"
    case_dtypes = ('uint64', 'uint64')


class BitShiftUint8Inputs2(_CpuBackendCaseBenchmark):
    case_prefix = "bitshift"
    case_dtypes = ('uint8', 'uint8')
