import numpy as np
import onnx
import onnx_ir
import onnxscript
from onnxscript import rewriter

import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light.onnx_core.optimization import (
    GraphBuilder as OptimizationGraphBuilder,
    GraphGraph,
    PatternOptimization as _PatternOptimization,
)

BLOCK_COUNTS = (50, 100, 250, 500)
FUSIONS_PER_BLOCK = 4
NODES_PER_BLOCK = 12
OPSET = 18


def fusion_blocks(op, value, bias, scale, weight, block_count):
    if block_count <= 0:
        raise ValueError("block_count must be positive.")
    for _ in range(block_count):
        value = op.Mul(op.Add(value, bias), scale)
        value = op.Add(op.MatMul(value, weight), bias)
        value = op.Relu(op.Add(value, bias))
        value = op.Log(op.Exp(op.Neg(op.Sqrt(op.Abs(op.Sub(value, bias))))))
    return value


def build_light(block_count):
    builder = GraphBuilder("fusion")
    builder.set_opset_version("", OPSET)
    value = builder.inp("X", TensorProto.FLOAT, ["N", 16])
    bias = builder.init(np.ones((1, 16), dtype=np.float32), name="bias")
    scale = builder.init(np.array(0.5, dtype=np.float32), name="scale")
    weight = builder.init(np.eye(16, dtype=np.float32), name="weight")
    value = fusion_blocks(builder.op, value, bias, scale, weight, block_count)
    builder.out(value, TensorProto.FLOAT, ["N", 16])
    return builder.to_onnx("model")


def build_onnxscript(block_count):
    graph = onnx_ir.Graph(
        inputs=[],
        outputs=[],
        nodes=[],
        opset_imports={"": OPSET},
        name="fusion",
    )
    builder = onnxscript.GraphBuilder(graph)
    value = builder.input("X", dtype=onnx_ir.DataType.FLOAT, shape=["N", 16])
    bias = builder.initializer(
        onnx_ir.tensor(np.ones((1, 16), dtype=np.float32)),
        name="bias",
    )
    scale = builder.initializer(
        onnx_ir.tensor(np.array(0.5, dtype=np.float32)),
        name="scale",
    )
    weight = builder.initializer(
        onnx_ir.tensor(np.eye(16, dtype=np.float32)),
        name="weight",
    )
    value = fusion_blocks(builder.op, value, bias, scale, weight, block_count)
    value.type = onnx_ir.TensorType(onnx_ir.DataType.FLOAT)
    value.shape = onnx_ir.Shape(["N", 16])
    builder.add_output(value, None)
    return onnx_ir.to_proto(onnx_ir.Model(graph, ir_version=onnx.IR_VERSION))


class _ChainFusionPattern(_PatternOptimization):
    def __init__(self, name, op_types, fused_op, replacement_inputs):
        super().__init__(priority=1, name=name)
        self.op_types = op_types
        self.fused_op = fused_op
        self.replacement_inputs = replacement_inputs

    def fast_op_type(self):
        return {self.op_types[-1]}

    def match(self, graph, node):
        matched = [node]
        current = node
        for op_type in reversed(self.op_types[:-1]):
            previous = graph.node_before(current.input[0])
            if previous is None or previous.op_type != op_type:
                return self.no_match(node, f"the chain does not contain {op_type}")
            if len(graph.next_nodes(previous.output[0])) != 1:
                return self.no_match(
                    node, f"the {op_type} output has multiple consumers"
                )
            matched.append(previous)
            current = previous
        matched.reverse()
        return self.result(matched, insert_at=node)

    def apply(self, graph, nodes):
        del graph
        inputs = [
            nodes[node_index].input[input_index]
            for node_index, input_index in self.replacement_inputs
        ]
        return [
            oh.make_node(
                self.fused_op,
                inputs,
                list(nodes[-1].output),
                domain="onnx_light.benchmark",
            )
        ]


def light_patterns():
    return [
        _ChainFusionPattern(
            "AddMulFusion",
            ("Add", "Mul"),
            "FusedAddMul",
            ((0, 0), (0, 1), (1, 1)),
        ),
        _ChainFusionPattern(
            "MatMulAddFusion",
            ("MatMul", "Add"),
            "FusedMatMulAdd",
            ((0, 0), (0, 1), (1, 1)),
        ),
        _ChainFusionPattern(
            "AddReluFusion",
            ("Add", "Relu"),
            "FusedAddRelu",
            ((0, 0), (0, 1)),
        ),
        _ChainFusionPattern(
            "SixNodeFusion",
            ("Sub", "Abs", "Sqrt", "Neg", "Exp", "Log"),
            "FusedSixNode",
            ((0, 0), (0, 1)),
        ),
    ]


def _add_mul_pattern(op, x, bias, scale):
    return op.Mul(op.Add(x, bias), scale)


def _fused_add_mul(op, x, bias, scale):
    return op.FusedAddMul(
        x,
        bias,
        scale,
        _domain="onnx_light.benchmark",
        _version=1,
    )


def _matmul_add_pattern(op, x, weight, bias):
    return op.Add(op.MatMul(x, weight), bias)


def _fused_matmul_add(op, x, weight, bias):
    return op.FusedMatMulAdd(
        x,
        weight,
        bias,
        _domain="onnx_light.benchmark",
        _version=1,
    )


def _add_relu_pattern(op, x, bias):
    return op.Relu(op.Add(x, bias))


def _fused_add_relu(op, x, bias):
    return op.FusedAddRelu(
        x,
        bias,
        _domain="onnx_light.benchmark",
        _version=1,
    )


def _six_node_pattern(op, x, bias):
    return op.Log(op.Exp(op.Neg(op.Sqrt(op.Abs(op.Sub(x, bias))))))


def _fused_six_node(op, x, bias):
    return op.FusedSixNode(
        x,
        bias,
        _domain="onnx_light.benchmark",
        _version=1,
    )


ONNXSCRIPT_RULES = rewriter.RewriteRuleSet(
    [
        rewriter.RewriteRule(
            _add_mul_pattern,
            _fused_add_mul,
            name="AddMulFusion",
        ),
        rewriter.RewriteRule(
            _matmul_add_pattern,
            _fused_matmul_add,
            name="MatMulAddFusion",
        ),
        rewriter.RewriteRule(
            _add_relu_pattern,
            _fused_add_relu,
            name="AddReluFusion",
        ),
        rewriter.RewriteRule(
            _six_node_pattern,
            _fused_six_node,
            name="SixNodeFusion",
        ),
    ]
)


class PatternFusion:
    params = (BLOCK_COUNTS, ("float32",), ("onnx-light", "onnxscript"))
    param_names = ("blocks", "dtype", "implementation")
    number = 1
    repeat = 1
    warmup_time = 0
    timeout = 120

    def setup(self, blocks, dtype, implementation):
        if blocks not in BLOCK_COUNTS or dtype != "float32":
            raise ValueError(f"Unexpected parameters {blocks!r}, {dtype!r}.")
        if implementation == "onnx-light":
            self.build = build_light
        elif implementation == "onnxscript":
            self.build = build_onnxscript
        else:
            raise ValueError(f"Unexpected implementation {implementation!r}.")
        model = self.build(blocks)
        self.expected = blocks * FUSIONS_PER_BLOCK
        self.fusion_graph = (
            OptimizationGraphBuilder(model)
            if implementation == "onnx-light"
            else onnx_ir.from_proto(model)
        )

    def time_construction(self, blocks, dtype, implementation):
        self.build(blocks)

    def time_fusion(self, blocks, dtype, implementation):
        if implementation == "onnx-light":
            count = len(GraphGraph(self.fusion_graph, light_patterns()).optimize())
        else:
            count = ONNXSCRIPT_RULES.apply_to_model(self.fusion_graph)
        if count != self.expected:
            raise AssertionError(f"Expected {self.expected} fusions, got {count}.")
