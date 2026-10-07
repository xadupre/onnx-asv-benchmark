from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MatMulInteger(_CpuBackendCaseBenchmark):
    case_prefix = "matmulinteger"
    case_names = (
        'test_cpu_matmulinteger_large_k_int8xint8_benchmark',
        'test_cpu_matmulinteger_large_k_int8xuint8_benchmark',
        'test_cpu_matmulinteger_large_k_uint8xint8_benchmark',
        'test_cpu_matmulinteger_large_k_uint8xuint8_benchmark',
        'test_cpu_matmulinteger_skinny_m_int8xint8_benchmark',
        'test_cpu_matmulinteger_skinny_m_int8xuint8_benchmark',
        'test_cpu_matmulinteger_skinny_m_uint8xint8_benchmark',
        'test_cpu_matmulinteger_skinny_m_uint8xuint8_benchmark',
        'test_cpu_matmulinteger_square_128_int8xint8_benchmark',
        'test_cpu_matmulinteger_square_128_int8xuint8_benchmark',
        'test_cpu_matmulinteger_square_128_uint8xint8_benchmark',
        'test_cpu_matmulinteger_square_128_uint8xuint8_benchmark',
        'test_cpu_matmulinteger_square_512_int8xint8_benchmark',
        'test_cpu_matmulinteger_square_512_int8xuint8_benchmark',
        'test_cpu_matmulinteger_square_512_uint8xint8_benchmark',
        'test_cpu_matmulinteger_square_512_uint8xuint8_benchmark',
        'test_cpu_matmulinteger_square_64_int8xint8_benchmark',
        'test_cpu_matmulinteger_square_64_int8xuint8_benchmark',
        'test_cpu_matmulinteger_square_64_uint8xint8_benchmark',
        'test_cpu_matmulinteger_square_64_uint8xuint8_benchmark',
    )
