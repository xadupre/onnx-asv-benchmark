from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class BatchNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "batchnormalization"
    case_names = (
        'test_cpu_batchnormalization_n2_c16_d4_h8_w8_rank5_float32_benchmark',
        'test_cpu_batchnormalization_n32_c128_rank2_float32_benchmark',
        'test_cpu_batchnormalization_n4_c16_h8_w8_rank4_bfloat16_benchmark',
        'test_cpu_batchnormalization_n4_c16_h8_w8_rank4_float16_benchmark',
        'test_cpu_batchnormalization_n4_c32_h16_w16_rank4_float32_benchmark',
        'test_cpu_batchnormalization_n8_c16_l64_rank3_float32_benchmark',
        'test_cpu_batchnormalization_n8_c64_h8_w8_rank4_float32_benchmark',
        'test_cpu_batchnormalization_training_n4_c32_h8_w8_rank4_float32_benchmark',
    )
