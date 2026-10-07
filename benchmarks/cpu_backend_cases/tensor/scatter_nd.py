from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class ScatterND(_CpuBackendCaseBenchmark):
    case_prefix = "scatternd"
    case_names = (
        'test_cpu_scatternd_hidden6656_one_row_indices32_bfloat16_benchmark',
        'test_cpu_scatternd_hidden6656_one_row_indices32_float16_benchmark',
        'test_cpu_scatternd_hidden6656_one_row_indices32_float32_benchmark',
        'test_cpu_scatternd_hidden6656_one_row_indices64_bfloat16_benchmark',
        'test_cpu_scatternd_hidden6656_one_row_indices64_float16_benchmark',
        'test_cpu_scatternd_hidden6656_one_row_indices64_float32_benchmark',
        'test_cpu_scatternd_hidden6656_rows_indices32_bfloat16_benchmark',
        'test_cpu_scatternd_hidden6656_rows_indices32_float16_benchmark',
        'test_cpu_scatternd_hidden6656_rows_indices32_float32_benchmark',
        'test_cpu_scatternd_hidden6656_rows_indices64_bfloat16_benchmark',
        'test_cpu_scatternd_hidden6656_rows_indices64_float16_benchmark',
        'test_cpu_scatternd_hidden6656_rows_indices64_float32_benchmark',
        'test_cpu_scatternd_small_rows_indices32_bfloat16_benchmark',
        'test_cpu_scatternd_small_rows_indices32_float16_benchmark',
        'test_cpu_scatternd_small_rows_indices32_float32_benchmark',
        'test_cpu_scatternd_small_rows_indices64_bfloat16_benchmark',
        'test_cpu_scatternd_small_rows_indices64_float16_benchmark',
        'test_cpu_scatternd_small_rows_indices64_float32_benchmark',
    )
