import unittest

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from benchmarks.onnx.builder.graph_builder import (
    GraphBuilderAttention,
    LARGE_INITIALIZER_COUNT,
    build_light,
    build_onnxscript,
)


class TestGraphBuilder(unittest.TestCase):
    def test_equivalent_models(self):
        rng = np.random.default_rng(0)
        for count in (20, 100):
            light = onnx.load_from_string(build_light(count).SerializeToString())
            scripted = build_onnxscript(count)
            for opset in light.opset_import:
                if opset.domain == "ai.onnx":
                    opset.domain = ""
            for model in (light, scripted):
                onnx.checker.check_model(model)
                self.assertEqual(len(model.graph.node), count)
            light_session = ReferenceEvaluator(light)
            scripted_session = ReferenceEvaluator(scripted)
            for shape in ((1, 1, 4), (2, 3, 6), (3, 5, 8)):
                with self.subTest(nodes=count, shape=shape):
                    feeds = {"X": rng.standard_normal(shape).astype(np.float32)}
                    actual = light_session.run(None, feeds)[0]
                    expected = scripted_session.run(None, feeds)[0]
                    self.assertEqual(actual.shape, shape)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_serialized_payload(self):
        payload = np.zeros(128, dtype=np.uint8)
        for build in (build_light, build_onnxscript):
            with self.subTest(builder=build.__name__):
                model = build(20, payload)
                for opset in model.opset_import:
                    if opset.domain == "ai.onnx":
                        opset.domain = ""
                onnx.checker.check_model(model)
                self.assertEqual(
                    len(model.graph.output),
                    LARGE_INITIALIZER_COUNT + 1,
                )
                serialized = model.SerializeToString()
                self.assertGreaterEqual(
                    len(serialized),
                    LARGE_INITIALIZER_COUNT * len(payload),
                )

    def test_benchmark_cases(self):
        self.assertEqual(
            GraphBuilderAttention.param_names,
            ("nodes", "dtype", "builder", "serialize"),
        )
        for builder in GraphBuilderAttention.params[2]:
            for serialize in GraphBuilderAttention.params[3]:
                with self.subTest(builder=builder, serialize=serialize):
                    benchmark = GraphBuilderAttention()
                    benchmark.setup(100, "float32", builder, serialize)
                    benchmark.time_build(100, "float32", builder, serialize)

    def test_invalid_node_count(self):
        for build in (build_light, build_onnxscript):
            with self.subTest(builder=build.__name__):
                with self.assertRaises(ValueError):
                    build(21)
