"""
Debug Conv2D bias extraction
"""
import onnx
import numpy as np
import sys
import os
sys.path.insert(0, 'cora_python')

from cora_python.g.macros.CORAROOT import CORAROOT

model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')

print(f"Loading ONNX model: {model_path}")
model = onnx.load(model_path)

initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

print(f"\nInitializers:")
for name, arr in initializers.items():
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

print(f"\nConv nodes:")
for i, node in enumerate(model.graph.node):
    if node.op_type == 'Conv':
        print(f"\nNode {i}: {node.op_type} - {node.name}")
        print(f"  Inputs: {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")
        
        if len(node.input) >= 2:
            weight_name = node.input[1]
            if weight_name in initializers:
                w = initializers[weight_name]
                print(f"  Weight '{weight_name}': shape={w.shape}")
        
        if len(node.input) >= 3:
            bias_name = node.input[2]
            if bias_name in initializers:
                b = initializers[bias_name]
                print(f"  Bias '{bias_name}': shape={b.shape}, values={b[:5] if b.size > 5 else b}")
            else:
                print(f"  Bias '{bias_name}': NOT FOUND in initializers")
        else:
            print(f"  No bias (only {len(node.input)} inputs)")

