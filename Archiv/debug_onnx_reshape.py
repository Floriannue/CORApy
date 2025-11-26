import sys
sys.path.insert(0, 'cora_python')

import onnx
import os
from cora_python.g.macros.CORAROOT import CORAROOT

model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')
print(f"Loading ONNX model: {model_path}")

model = onnx.load(model_path)
graph = model.graph

print("\nONNX Graph Nodes:")
for i, node in enumerate(graph.node):
    print(f"\nNode {i}: {node.op_type} - {node.name}")
    print(f"  Inputs: {list(node.input)}")
    print(f"  Outputs: {list(node.output)}")
    
    if node.op_type == 'Reshape':
        print(f"  Reshape node found!")
        for attr in node.attribute:
            print(f"    Attribute {attr.name}: {list(attr.ints) if attr.ints else attr.i}")
        
        # Check if shape is in initializers
        if len(node.input) >= 2:
            shape_name = node.input[1]
            initializers = {init.name: init for init in graph.initializer}
            if shape_name in initializers:
                shape_tensor = onnx.numpy_helper.to_array(initializers[shape_name])
                print(f"    Shape from initializer: {shape_tensor}")
    
    if node.op_type == 'Gemm' or node.op_type == 'MatMul':
        print(f"  FullyConnected/MatMul node found!")
        # Check weights
        if len(node.input) >= 2:
            weight_name = node.input[1]
            initializers = {init.name: init for init in graph.initializer}
            if weight_name in initializers:
                weight = onnx.numpy_helper.to_array(initializers[weight_name])
                print(f"    Weight shape: {weight.shape}")

