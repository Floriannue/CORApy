"""Debug script to check Pad operation before AveragePool"""
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

print("ONNX Model Info:")
print("=" * 80)
print(f"IR Version: {model.ir_version}")
print(f"Opset Version: {model.opset_import[0].version if model.opset_import else 'Unknown'}")
print(f"Producer: {model.producer_name} {model.producer_version}")

print("\n\nONNX Model Graph Nodes (around Pad and AveragePool):")
print("=" * 80)
initializers = {init.name: init for init in graph.initializer}
print(f"\nAll initializers: {list(initializers.keys())}\n")

for i, node in enumerate(graph.node):
    if node.op_type in ['Pad', 'AveragePool']:
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
            elif attr.type == 4:  # STRING
                print(f"    {attr.name}: {attr.s.decode('utf-8')}")
        
        # Check if pads/value are provided as input tensors (ONNX opset 11+)
        if node.op_type == 'Pad':
            print(f"  Checking Pad inputs:")
            for j, inp in enumerate(node.input):
                print(f"    Input {j}: {inp}")
                if inp in initializers:
                    val = onnx.numpy_helper.to_array(initializers[inp])
                    print(f"      Value: {val}")
                else:
                    print(f"      (not in initializers)")
        
        # Check input/output shapes from value_info
        print(f"  Shape info:")
        input_shapes = {}
        output_shapes = {}
        for value_info in graph.value_info:
            shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in value_info.type.tensor_type.shape.dim]
            if value_info.name in node.input:
                input_shapes[value_info.name] = shape
                print(f"    Input {value_info.name}: {shape}")
            if value_info.name in node.output:
                output_shapes[value_info.name] = shape
                print(f"    Output {value_info.name}: {shape}")
        
        # Also check graph input/output
        for inp in graph.input:
            if inp.name in node.input:
                shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in inp.type.tensor_type.shape.dim]
                input_shapes[inp.name] = shape
                print(f"    Input {inp.name} (graph input): {shape}")
        for out in graph.output:
            if out.name in node.output:
                shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in out.type.tensor_type.shape.dim]
                output_shapes[out.name] = shape
                print(f"    Output {out.name} (graph output): {shape}")
        
        # For Pad, try to infer padding from shapes
        if node.op_type == 'Pad' and len(node.input) > 0 and len(node.output) > 0:
            inp_name = node.input[0]
            out_name = node.output[0]
            if inp_name in input_shapes and out_name in output_shapes:
                inp_shape = input_shapes[inp_name]
                out_shape = output_shapes[out_name]
                if len(inp_shape) >= 2 and len(out_shape) >= 2 and all(isinstance(s, int) for s in inp_shape + out_shape):
                    # For 4D tensor [N, C, H, W], compare H and W
                    if len(inp_shape) == 4 and len(out_shape) == 4:
                        h_pad = out_shape[2] - inp_shape[2]
                        w_pad = out_shape[3] - inp_shape[3]
                        print(f"  Inferred padding (from shapes): H={h_pad}, W={w_pad}")
                        if h_pad > 0 or w_pad > 0:
                            # Assume symmetric padding
                            top = h_pad // 2
                            bottom = h_pad - top
                            left = w_pad // 2
                            right = w_pad - left
                            print(f"    Inferred pads: [top={top}, left={left}, bottom={bottom}, right={right}]")
                            print(f"    Or as list: [{top}, {left}, {bottom}, {right}]")

