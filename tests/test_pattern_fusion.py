import unittest

from benchmarks.builder.pattern.pattern_fusion import (
    BLOCK_COUNTS,
    FUSIONS_PER_BLOCK,
    NODES_PER_BLOCK,
    PatternFusion,
    build_light,
    build_onnxscript,
    light_patterns,
)


class TestPatternFusion(unittest.TestCase):
    def test_equivalent_models(self):
        light = build_light(2)
        scripted = build_onnxscript(2)
        self.assertEqual(
            [node.op_type for node in light.graph.node],
            [node.op_type for node in scripted.graph.node],
        )
        self.assertEqual(len(light.graph.node), 2 * NODES_PER_BLOCK)

    def test_patterns(self):
        patterns = light_patterns()
        self.assertEqual(len(patterns), FUSIONS_PER_BLOCK)
        six_node = next(
            pattern for pattern in patterns if pattern.name == "SixNodeFusion"
        )
        self.assertEqual(
            six_node.op_types,
            ("Sub", "Abs", "Sqrt", "Neg", "Exp", "Log"),
        )

    def test_benchmark_cases(self):
        self.assertEqual(
            PatternFusion.param_names,
            ("blocks", "dtype", "implementation"),
        )
        for blocks in BLOCK_COUNTS:
            for implementation in PatternFusion.params[2]:
                for method_name in ("time_construction", "time_fusion"):
                    with self.subTest(
                        blocks=blocks,
                        implementation=implementation,
                        method=method_name,
                    ):
                        benchmark = PatternFusion()
                        benchmark.setup(blocks, "float32", implementation)
                        getattr(benchmark, method_name)(
                            blocks, "float32", implementation
                        )
                        if method_name == "time_construction":
                            self.assertIsNotNone(benchmark.model)
                        benchmark.teardown(blocks, "float32", implementation)
                        self.assertIsNone(benchmark.model)
                        self.assertIsNone(benchmark.optimizer)
                        self.assertIsNone(benchmark.fusion_graph)

    def test_invalid_block_count(self):
        for build in (build_light, build_onnxscript):
            with self.subTest(builder=build.__name__):
                with self.assertRaises(ValueError):
                    build(0)


if __name__ == "__main__":
    unittest.main()
