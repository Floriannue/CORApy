"""Debug script to inspect the final layer of the ONNX model"""
import onnx
import numpy as np
import os
import sys

# Find model path
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and not os.path.exists(os.path.join(current_dir, 'cora_matlab')):
    current_dir = os.path.dirname(current_dir)

model_path = os.path.join(current_dir, 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')

model = onnx.load(model_path)
graph = model.graph

print("ONNX Model Graph Nodes (last 5):")
print("=" * 80)
for i, node in enumerate(graph.node[-5:]):
    actual_idx = len(graph.node) - 5 + i
    print(f"\nNode {actual_idx}: {node.op_type}")
    print(f"  Inputs: {list(node.input)}")
    print(f"  Outputs: {list(node.output)}")
    
    if node.op_type in ['Gemm', 'MatMul']:
        print(f"  {node.op_type} Attributes:")
        for attr in node.attribute:
            print(f"    {attr.name}: {attr.i if attr.type == 2 else attr.f}")
        
        # Get weight and bias
        initializers = {init.name: init for init in graph.initializer}
        if len(node.input) >= 2:
            weight_name = node.input[1]
            if weight_name in initializers:
                weight = onnx.numpy_helper.to_array(initializers[weight_name])
                print(f"  Weight shape: {weight.shape}")
                print(f"  Weight (first 3x3):\n{weight[:3, :3] if weight.ndim == 2 else weight[:3]}")
        
        if len(node.input) >= 3:
            bias_name = node.input[2]
            if bias_name in initializers:
                bias = onnx.numpy_helper.to_array(initializers[bias_name])
                print(f"  Bias shape: {bias.shape}")
                print(f"  Bias (first 5): {bias[:5] if len(bias) >= 5 else bias}")
        elif node.op_type == 'MatMul' and actual_idx + 1 < len(graph.node):
            # Check if next node is Add
            next_node = graph.node[actual_idx + 1]
            if next_node.op_type == 'Add':
                print(f"  Next node is Add, checking for bias...")
                if len(next_node.input) >= 2:
                    bias_name = next_node.input[1]
                    if bias_name in initializers:
                        bias = onnx.numpy_helper.to_array(initializers[bias_name])
                        print(f"  Bias shape: {bias.shape}")
                        print(f"  Bias (first 5): {bias[:5] if len(bias) >= 5 else bias}")

print("\n\nFinal output node:")
print(f"  Output: {graph.output[0].name}")
print(f"  Output shape: {[dim.dim_value if dim.dim_value > 0 else '?' for dim in graph.output[0].type.tensor_type.shape.dim]}")

