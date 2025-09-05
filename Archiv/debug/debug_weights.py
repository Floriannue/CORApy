import numpy as np
import onnx
import os

# Path to the ONNX model
model_path = "cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx"

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)

print("Loading ONNX model to check weight dimensions...")
model = onnx.load(model_path)

# Check the graph structure
graph = model.graph
print(f"Model has {len(graph.node)} nodes")
print(f"Model has {len(graph.initializer)} initializers")

# Find the first MatMul operation (fully connected layer)
matmul_nodes = [node for node in graph.node if node.op_type == 'MatMul']
if matmul_nodes:
    print(f"\nFound {len(matmul_nodes)} MatMul operations")
    
    for i, node in enumerate(matmul_nodes):
        print(f"\nMatMul node {i}: {node.name}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")
        
        # Check if weights are available
        if len(node.input) >= 2:
            weight_name = node.input[1]
            print(f"  Weight name: {weight_name}")
            
            # Find the weight in initializers
            weight_found = False
            for initializer in graph.initializer:
                if initializer.name == weight_name:
                    weight_array = onnx.numpy_helper.to_array(initializer)
                    print(f"  Weight shape: {weight_array.shape}")
                    print(f"  Weight dtype: {weight_array.dtype}")
                    print(f"  Weight first few values: {weight_array.flatten()[:10]}")
                    weight_found = True
                    break
            
            if not weight_found:
                print(f"  Weight not found in initializers")
        else:
            print(f"  No weight input found")

# Check input and output shapes
if graph.input:
    print(f"\nInput shapes:")
    for input_info in graph.input:
        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(dim.dim_param)
        print(f"  {input_info.name}: {shape}")

if graph.output:
    print(f"\nOutput shapes:")
    for output_info in graph.output:
        shape = []
        for dim in output_info.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(dim.dim_param)
        print(f"  {output_info.name}: {shape}")

print("\nNow let's check what the Python code actually loads...")
