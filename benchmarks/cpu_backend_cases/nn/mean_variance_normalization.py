from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class MeanVarianceNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "meanvariancenormalization"
    case_names = (
        'test_cpu_meanvariancenormalization_n2_c8_d4_h8_w8_axes0_2_3_4_rank5_float32_benchmark',
        'test_cpu_meanvariancenormalization_n4_c16_h32_w8_axes2_3_rank4_float32_benchmark',
        'test_cpu_meanvariancenormalization_n4_c16_h8_w8_default_rank4_bfloat16_benchmark',
        'test_cpu_meanvariancenormalization_n4_c16_h8_w8_default_rank4_float16_benchmark',
        'test_cpu_meanvariancenormalization_n8_c32_h16_w16_default_rank4_float32_benchmark',
        'test_cpu_meanvariancenormalization_n8_c32_l128_axes0_2_rank3_float32_benchmark',
        'test_cpu_meanvariancenormalization_r256_w128_axes1_rank2_float32_benchmark',
        'test_cpu_meanvariancenormalization_r64_w32_axes0_1_rank2_float32_benchmark',
    )
