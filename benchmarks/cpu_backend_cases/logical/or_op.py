from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class Or(_CpuBackendCaseBenchmark):
    case_prefix = "or"
    case_names = (
        'test_cpu_or_v7_contiguous_boolxbool_to_bool_n1048576_benchmark',
        'test_cpu_or_v7_contiguous_boolxbool_to_bool_n4096_benchmark',
        'test_cpu_or_v7_contiguous_boolxbool_to_bool_n4194304_benchmark',
        'test_cpu_or_v7_contiguous_boolxbool_to_bool_n65536_benchmark',
        'test_cpu_or_v7_general_boolxbool_to_bool_n1048576_benchmark',
        'test_cpu_or_v7_general_boolxbool_to_bool_n4096_benchmark',
        'test_cpu_or_v7_general_boolxbool_to_bool_n4194304_benchmark',
        'test_cpu_or_v7_general_boolxbool_to_bool_n65536_benchmark',
        'test_cpu_or_v7_left_scalar_boolxbool_to_bool_n1048576_benchmark',
        'test_cpu_or_v7_left_scalar_boolxbool_to_bool_n4096_benchmark',
        'test_cpu_or_v7_left_scalar_boolxbool_to_bool_n4194304_benchmark',
        'test_cpu_or_v7_left_scalar_boolxbool_to_bool_n65536_benchmark',
        'test_cpu_or_v7_outer_boolxbool_to_bool_n1048576_benchmark',
        'test_cpu_or_v7_outer_boolxbool_to_bool_n4096_benchmark',
        'test_cpu_or_v7_outer_boolxbool_to_bool_n4194304_benchmark',
        'test_cpu_or_v7_outer_boolxbool_to_bool_n65536_benchmark',
        'test_cpu_or_v7_per_channel_boolxbool_to_bool_n1048576_benchmark',
        'test_cpu_or_v7_per_channel_boolxbool_to_bool_n4096_benchmark',
        'test_cpu_or_v7_per_channel_boolxbool_to_bool_n4194304_benchmark',
        'test_cpu_or_v7_per_channel_boolxbool_to_bool_n65536_benchmark',
        'test_cpu_or_v7_right_scalar_boolxbool_to_bool_n1048576_benchmark',
        'test_cpu_or_v7_right_scalar_boolxbool_to_bool_n4096_benchmark',
        'test_cpu_or_v7_right_scalar_boolxbool_to_bool_n4194304_benchmark',
        'test_cpu_or_v7_right_scalar_boolxbool_to_bool_n65536_benchmark',
        'test_cpu_or_v7_row_boolxbool_to_bool_n1048576_benchmark',
        'test_cpu_or_v7_row_boolxbool_to_bool_n4096_benchmark',
        'test_cpu_or_v7_row_boolxbool_to_bool_n4194304_benchmark',
        'test_cpu_or_v7_row_boolxbool_to_bool_n65536_benchmark',
    )
