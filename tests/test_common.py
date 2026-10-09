import unittest

import numpy as np

from benchmarks.common import input_shape_label
from benchmarks.ops.traditionalml.cast_map import CastMap


class TestInputShapeLabel(unittest.TestCase):
    def test_map_input(self):
        self.assertEqual(input_shape_label({"x": {1: 2, 3: 4}}), "x=map[2->2]")
        self.assertEqual(
            input_shape_label({"x": {1: 2}, "y": np.zeros((2, 3))}),
            "x=map[1->1], y=2x3",
        )

    def test_tensor_and_empty_inputs(self):
        self.assertEqual(input_shape_label({"x": np.zeros(())}), "x=scalar")
        self.assertEqual(input_shape_label({}), "no inputs")

    def test_cast_map_benchmark(self):
        benchmark = CastMap()
        shape = benchmark.params[0][0]
        benchmark.setup(shape, "onnx-light")
        benchmark.time_run(shape, "onnx-light")
