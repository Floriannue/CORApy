#!/usr/bin/env python3
"""
Debug Flatten axis value
"""

import onnx

# Load the ONNX model
model_path = 'cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx'
model = onnx.load(model_path)

# Find the Flatten node
for i, node in enumerate(model.graph.node):
    if node.op_type == 'Flatten':
        print(f"Flatten node found at index {i}")
        print(f"  Inputs: {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")
        for attr in node.attribute:
            print(f"  Attribute {attr.name}: {attr.i}")
        break
