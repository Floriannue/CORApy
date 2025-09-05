#!/usr/bin/env python3
"""
Debug script to examine ONNX file structure
"""

import onnx
import numpy as np

def debug_onnx_file(file_path):
    """Debug ONNX file structure"""
    print(f"Loading ONNX file: {file_path}")
    
    # Load the model
    model = onnx.load(file_path)
    
    # Get the graph
    graph = model.graph
    
    print(f"\nModel inputs: {len(graph.input)}")
    for i, input_info in enumerate(graph.input):
        print(f"  Input {i}: {input_info.name}")
        print(f"    Shape: {[dim.dim_value for dim in input_info.type.tensor_type.shape.dim]}")
    
    print(f"\nModel outputs: {len(graph.output)}")
    for i, output_info in enumerate(graph.output):
        print(f"  Output {i}: {output_info.name}")
        print(f"    Shape: {[dim.dim_value for dim in output_info.type.tensor_type.shape.dim]}")
    
    print(f"\nInitializers: {len(graph.initializer)}")
    initializers = {init.name: init for init in graph.initializer}
    for name, init in initializers.items():
        print(f"  {name}: shape {list(init.dims)}, type {init.data_type}")
    
    print(f"\nNodes: {len(graph.node)}")
    for i, node in enumerate(graph.node):
        print(f"\n  Node {i}: {node.op_type}")
        print(f"    Name: {node.name if node.name else 'unnamed'}")
        print(f"    Inputs: {list(node.input)}")
        print(f"    Outputs: {list(node.output)}")
        print(f"    Attributes: {len(node.attribute)}")
        
        for attr in node.attribute:
            print(f"      {attr.name}: {attr.type}")
            if attr.type == onnx.AttributeProto.INTS:
                print(f"        Values: {list(attr.ints)}")
            elif attr.type == onnx.AttributeProto.INT:
                print(f"        Value: {attr.i}")
            elif attr.type == onnx.AttributeProto.FLOATS:
                print(f"        Values: {list(attr.floats)}")
            elif attr.type == onnx.AttributeProto.FLOAT:
                print(f"        Value: {attr.f}")
            elif attr.type == onnx.AttributeProto.STRING:
                print(f"        Value: {attr.s}")
        
        # Special handling for Reshape nodes
        if node.op_type == 'Reshape':
            print(f"    *** RESHAPE NODE DETAILS ***")
            print(f"    Input 0 (data): {node.input[0] if len(node.input) > 0 else 'None'}")
            print(f"    Input 1 (shape): {node.input[1] if len(node.input) > 1 else 'None'}")
            
            # Check if shape input is in initializers
            if len(node.input) > 1 and node.input[1] in initializers:
                shape_tensor = onnx.numpy_helper.to_array(initializers[node.input[1]])
                print(f"    Shape tensor value: {shape_tensor}")
                print(f"    Shape tensor shape: {shape_tensor.shape}")
                print(f"    Shape tensor dtype: {shape_tensor.dtype}")

if __name__ == "__main__":
    # Debug the ONNX file used in the example
    onnx_file = "cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx"
    debug_onnx_file(onnx_file)
