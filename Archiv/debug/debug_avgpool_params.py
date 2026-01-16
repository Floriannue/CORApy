"""Debug script to check AvgPool parameters from ONNX"""
import onnx
import numpy as np
import os

# Find model path
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and not os.path.exists(os.path.join(current_dir, 'cora_matlab')):
    current_dir = os.path.dirname(current_dir)

model_path = os.path.join(current_dir, 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')

model = onnx.load(model_path)
graph = model.graph

print("ONNX Model Graph Nodes:")
print("=" * 80)
for i, node in enumerate(graph.node):
    if node.op_type == 'AveragePool':
        print(f"\nNode {i}: {node.op_type}")
        print(f"  Inputs: {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")
        print(f"  Attributes:")
        for attr in node.attribute:
            if attr.type == 2:  # INT
                print(f"    {attr.name}: {attr.i}")
            elif attr.type == 3:  # INTS
                print(f"    {attr.name}: {list(attr.ints)}")
            elif attr.type == 1:  # FLOAT
                print(f"    {attr.name}: {attr.f}")
            elif attr.type == 6:  # FLOATS
                print(f"    {attr.name}: {list(attr.floats)}")

