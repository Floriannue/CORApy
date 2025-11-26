"""
Check what operations are in the ONNX model
"""
import onnx
import numpy as np
import sys
sys.path.insert(0, 'cora_python')
from cora_python.g.macros.CORAROOT import CORAROOT
import os

model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'nn-nav-set.onnx')
model = onnx.load(model_path)

print("=== All Operations in ONNX Model ===")
for i, node in enumerate(model.graph.node):
    print(f"{i}. {node.op_type} - {node.name}")
    print(f"   Inputs: {list(node.input)}")
    print(f"   Outputs: {list(node.output)}")

print(f"\n=== Initializers (Weights/Biases) ===")
initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}
for name, arr in initializers.items():
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")

