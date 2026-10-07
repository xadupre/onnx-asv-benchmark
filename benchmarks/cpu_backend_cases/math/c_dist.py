from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class CDist(_CpuBackendCaseBenchmark):
    case_prefix = "cdist"
    case_names = (
        'test_cpu_cdist_m256_k128_n64_euclidean_float32_benchmark',
        'test_cpu_cdist_m256_k128_n64_euclidean_float64_benchmark',
        'test_cpu_cdist_m256_k128_n64_sqeuclidean_float32_benchmark',
        'test_cpu_cdist_m256_k128_n64_sqeuclidean_float64_benchmark',
        'test_cpu_cdist_m512_k512_n128_euclidean_float32_benchmark',
        'test_cpu_cdist_m512_k512_n128_euclidean_float64_benchmark',
        'test_cpu_cdist_m512_k512_n128_sqeuclidean_float32_benchmark',
        'test_cpu_cdist_m512_k512_n128_sqeuclidean_float64_benchmark',
        'test_cpu_cdist_m64_k64_n64_euclidean_float32_benchmark',
        'test_cpu_cdist_m64_k64_n64_euclidean_float64_benchmark',
        'test_cpu_cdist_m64_k64_n64_sqeuclidean_float32_benchmark',
        'test_cpu_cdist_m64_k64_n64_sqeuclidean_float64_benchmark',
    )
