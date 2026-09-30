import unittest

import numpy as np

from benchmarks.common import BACKENDS
from benchmarks.maths.add import Add
from benchmarks.maths.matmul import MatMul
from benchmarks.models.matmul_add import MatMulAdd
from benchmarks.models.mlp import MLP
from benchmarks.nn.relu import Relu


class TestBenchmarks(unittest.TestCase):
    def test_setup_and_run(self):
        for benchmark_type, output_shape in (
            (Add, (1024, 1024)),
            (MatMul, (256, 256)),
            (MatMulAdd, (64, 256)),
            (MLP, (32, 64)),
            (Relu, (1024, 1024)),
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
                    self.assertEqual(outputs[0].dtype, np.float32)
                    self.assertTrue(np.isfinite(outputs[0]).all())


if __name__ == "__main__":
    unittest.main()
