import unittest

import numpy as np

from benchmarks.common import BACKENDS
from benchmarks.logical.and_op import And
from benchmarks.math.add import Add
from benchmarks.math.binary import Binary, OPERATORS
from benchmarks.math.gemm import Gemm
from benchmarks.math.matmul import MatMul
from benchmarks.math.reciprocal import Reciprocal
from benchmarks.math.reduce_sum import ReduceSum
from benchmarks.math.where import Where
from benchmarks.models.matmul_add import MatMulAdd
from benchmarks.models.mlp import MLP
from benchmarks.nn.affine_grid import AffineGrid
from benchmarks.nn.conv import Conv
from benchmarks.nn.gru import Gru
from benchmarks.nn.relu import Relu
from benchmarks.nn.rms_normalization import RMSNormalization


class TestBenchmarks(unittest.TestCase):
    def test_setup_and_run(self):
        for benchmark_type, output_shape in (
            (Add, (1024, 1024)),
            (And, (512, 512)),
            (Gemm, (64, 256)),
            (MatMul, (256, 256)),
            (Reciprocal, (1024, 1024)),
            (ReduceSum, (64, 64)),
            (Where, (512, 512)),
            (MatMulAdd, (64, 256)),
            (MLP, (32, 64)),
            (AffineGrid, (1, 32, 32, 2)),
            (Conv, (1, 16, 30, 30)),
            (Gru, (8, 1, 4, 32)),
            (Relu, (1024, 1024)),
            (RMSNormalization, (32, 256)),
        ):
            for backend in BACKENDS:
                with self.subTest(benchmark=benchmark_type.__name__, backend=backend):
                    benchmark = benchmark_type()
                    self.assertIn(backend, benchmark.params)
                    benchmark.setup(backend)
                    benchmark.time_run(backend)
                    outputs = benchmark.session.run(None, benchmark.feeds)
                    self.assertEqual(len(outputs), 1)
                    self.assertEqual(outputs[0].shape, output_shape)
                    self.assertEqual(
                        outputs[0].dtype,
                        np.bool_ if benchmark_type is And else np.float32,
                    )
                    self.assertTrue(np.isfinite(outputs[0]).all())

    def test_binary_operators(self):
        self.assertEqual(
            set(OPERATORS),
            {
                "Sub", "Mul", "Div", "Pow", "Mod", "Min", "Max", "Mean",
                "Sum", "Equal", "Greater", "GreaterOrEqual", "Less",
                "LessOrEqual", "And", "Or", "Xor", "BitwiseAnd",
                "BitwiseOr", "BitwiseXor", "BitShift", "PRelu",
            },
        )
        for operator in OPERATORS:
            for backend in BACKENDS:
                with self.subTest(operator=operator, backend=backend):
                    benchmark = Binary()
                    self.assertIn(operator, benchmark.params[0])
                    self.assertIn(backend, benchmark.params[1])
                    benchmark.setup(operator, backend)
                    benchmark.time_run(operator, backend)
                    outputs = benchmark.session.run(None, benchmark.feeds)
                    self.assertEqual(len(outputs), 1)
                    self.assertEqual(outputs[0].shape, (1024, 1024))
                    if operator in (
                        "Equal", "Greater", "GreaterOrEqual", "Less",
                        "LessOrEqual", "And", "Or", "Xor",
                    ):
                        self.assertEqual(outputs[0].dtype, np.bool_)
                    elif operator.startswith("Bitwise") or operator == "BitShift":
                        self.assertEqual(outputs[0].dtype, np.uint32)
                    else:
                        self.assertEqual(outputs[0].dtype, np.float32)
                        self.assertTrue(np.isfinite(outputs[0]).all())


if __name__ == "__main__":
    unittest.main()
