from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class InstanceNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "instancenormalization"
    case_names = (
        'test_cpu_instancenormalization_n2_c16_d4_h8_w8_rank5_float32_benchmark',
        'test_cpu_instancenormalization_n2_c16_h8_w8_rank4_bfloat16_benchmark',
        'test_cpu_instancenormalization_n2_c16_h8_w8_rank4_float16_benchmark',
        'test_cpu_instancenormalization_n2_c8_h64_w8_rank4_float32_benchmark',
        'test_cpu_instancenormalization_n4_c32_h16_w16_rank4_float32_benchmark',
        'test_cpu_instancenormalization_n8_c16_l128_rank3_float32_benchmark',
    )
