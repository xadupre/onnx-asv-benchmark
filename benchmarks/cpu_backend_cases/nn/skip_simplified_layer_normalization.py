from benchmarks.cpu_backend_cases._base import (
    _CpuBackendCaseBenchmark,
)


class SkipSimplifiedLayerNormalization(_CpuBackendCaseBenchmark):
    case_prefix = "skip"
    case_names = (
        'test_cpu_skip_simplified_layer_normalization_decode_1x1x4096_bfloat16_benchmark',
        'test_cpu_skip_simplified_layer_normalization_decode_1x1x4096_float16_benchmark',
        'test_cpu_skip_simplified_layer_normalization_decode_1x1x4096_float32_benchmark',
        'test_cpu_skip_simplified_layer_normalization_prefill_1x128x4096_bfloat16_benchmark',
        'test_cpu_skip_simplified_layer_normalization_prefill_1x128x4096_float16_benchmark',
        'test_cpu_skip_simplified_layer_normalization_prefill_1x128x4096_float32_benchmark',
    )
