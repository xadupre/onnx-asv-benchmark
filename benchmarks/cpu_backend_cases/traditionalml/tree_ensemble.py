from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class TreeEnsemble(_CpuBackendCaseBenchmark):
    case_prefix = "treeensemble"
    case_names = (
        'test_cpu_treeensemble_t10000_f4096_b128_float32_benchmark',
        'test_cpu_treeensemble_t10000_f4096_b1_float32_benchmark',
        'test_cpu_treeensemble_t1000_f1024_b32_float32_benchmark',
        'test_cpu_treeensemble_t100_f64_b8_float32_benchmark',
        'test_cpu_treeensemble_t10_f4_b1_float32_benchmark',
    )
