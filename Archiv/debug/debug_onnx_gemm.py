"""Debug script to inspect ONNX Gemm operations"""
import onnx
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

# Find CORAROOT by looking for cora_matlab directory
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and not os.path.exists(os.path.join(current_dir, 'cora_matlab')):
    current_dir = os.path.dirname(current_dir)

model_path = os.path.join(current_dir, 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')

model = onnx.load(model_path)
graph = model.graph

print("ONNX Model Graph Nodes:")
print("=" * 80)
for i, node in enumerate(graph.node):
    print(f"\nNode {i}: {node.op_type}")
    print(f"  Inputs: {list(node.input)}")
    print(f"  Outputs: {list(node.output)}")
    
    if node.op_type == 'Gemm':
        print("  Gemm Attributes:")
        for attr in node.attribute:
            print(f"    {attr.name}: {attr.i if attr.type == 2 else attr.f}")
        
        # Get weight and bias
        initializers = {init.name: init for init in graph.initializer}
        if len(node.input) >= 2:
            weight_name = node.input[1]
            if weight_name in initializers:
                weight = onnx.numpy_helper.to_array(initializers[weight_name])
                print(f"  Weight shape: {weight.shape}")
                print(f"  Weight (first 5x5):\n{weight[:5, :5] if weight.ndim == 2 else weight[:5]}")
        
        if len(node.input) >= 3:
            bias_name = node.input[2]
            if bias_name in initializers:
                bias = onnx.numpy_helper.to_array(initializers[bias_name])
                print(f"  Bias shape: {bias.shape}")
                print(f"  Bias (first 5): {bias[:5] if len(bias) >= 5 else bias}")

