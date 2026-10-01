import numpy as np
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto

from benchmarks.common import BACKENDS, run_session, setup_session

OPERATORS = (
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Mod",
    "Min",
    "Max",
    "Mean",
    "Sum",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "And",
    "Or",
    "Xor",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "BitShift",
    "PRelu",
)


class Binary:
    params = (OPERATORS, BACKENDS)
    param_names = ("operator", "backend")
    number = 4
    timeout = 10

    def setup(self, operator, backend):
        shape = (1024, 1024)
        rng = np.random.default_rng(2)
        if operator in ("And", "Or", "Xor"):
            tensor_type = output_type = TensorProto.BOOL
            x = rng.integers(0, 2, size=shape).astype(np.bool_)
            y = rng.integers(0, 2, size=shape).astype(np.bool_)
        elif operator.startswith("Bitwise") or operator == "BitShift":
            tensor_type = output_type = TensorProto.UINT32
            x = rng.integers(0, 256, size=shape, dtype=np.uint32)
            y = rng.integers(0, 8, size=shape, dtype=np.uint32)
        else:
            tensor_type = TensorProto.FLOAT
            output_type = (
                TensorProto.BOOL
                if operator in ("Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual")
                else TensorProto.FLOAT
            )
            x = rng.standard_normal(shape, dtype=np.float32)
            y = rng.standard_normal(shape, dtype=np.float32)
            if operator in ("Div", "Mod"):
                y = rng.uniform(0.5, 2.0, size=shape).astype(np.float32)
            elif operator == "Pow":
                x = np.abs(x) + np.float32(0.1)

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        operator,
                        ["X", "Y"],
                        ["Z"],
                        **(
                            {"direction": "LEFT"}
                            if operator == "BitShift"
                            else {"fmod": 1} if operator == "Mod" else {}
                        ),
                    )
                ],
                operator.lower(),
                [
                    oh.make_tensor_value_info("X", tensor_type, shape),
                    oh.make_tensor_value_info("Y", tensor_type, shape),
                ],
                [oh.make_tensor_value_info("Z", output_type, shape)],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        setup_session(self, backend, model, {"X": x, "Y": y})

    def time_run(self, operator, backend):
        run_session(self)
