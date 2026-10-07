from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class GroupNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "groupnormalization"
    case_names = (
        'test_cpu_groupnormalization_n1_c64_g32_h8_w8_rank4_float32_benchmark',
        'test_cpu_groupnormalization_n2_c16_g4_h8_w8_rank4_bfloat16_benchmark',
        'test_cpu_groupnormalization_n2_c16_g4_h8_w8_rank4_float16_benchmark',
        'test_cpu_groupnormalization_n2_c16_g4_l64_rank3_float32_benchmark',
        'test_cpu_groupnormalization_n2_c24_g6_d4_h8_w8_rank5_float32_benchmark',
        'test_cpu_groupnormalization_n4_c32_g8_h16_w16_rank4_float32_benchmark',
    )
