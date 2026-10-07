from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class NonZero(_CpuBackendCaseBenchmark):
    case_prefix = "nonzero"
    case_names = (
        'test_cpu_nonzero_tokens_all_bool_benchmark',
        'test_cpu_nonzero_tokens_mixed_bool_benchmark',
        'test_cpu_nonzero_tokens_zero_bool_benchmark',
    )
