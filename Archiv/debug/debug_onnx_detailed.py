"""
Detailed ONNX model inspection
"""
import numpy as np
import os
import onnx
import onnxruntime as ort
import sys
sys.path.insert(0, 'cora_python')

from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.nn.neuralNetwork import NeuralNetwork

model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'nn-nav-set.onnx')
model = onnx.load(model_path)

print("=== ONNX Model Structure ===")
initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

print(f"\nAll initializers:")
for name, arr in initializers.items():
    print(f"  {name}: shape={arr.shape}")

print(f"\nGraph nodes in order:")
for i, node in enumerate(model.graph.node):
    print(f"\n{i}. {node.op_type} - {node.name}")
    print(f"   Inputs: {node.input}")
    print(f"   Outputs: {node.output}")
    
    if node.op_type in ['Gemm', 'MatMul']:
        for inp in node.input:
            if inp in initializers:
                w = initializers[inp]
                print(f"   Weight '{inp}': shape={w.shape}")
                if w.size <= 10:
                    print(f"     values: {w}")
        
        # Check for bias
        if len(node.input) >= 3 and node.input[2] in initializers:
            b = initializers[node.input[2]]
            print(f"   Bias '{node.input[2]}': shape={b.shape}")
            if b.size <= 10:
                print(f"     values: {b}")
    
    # Check attributes
    if node.attribute:
        print(f"   Attributes:")
        for attr in node.attribute:
            val = onnx.helper.get_attribute_value(attr)
            print(f"     {attr.name}: {val}")

# Test with ONNX Runtime
print(f"\n=== ONNX Runtime Test ===")
sess = ort.InferenceSession(model_path)
x = np.ones((1, 4), dtype=np.float32)
y_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0]
print(f"Input: {x}")
print(f"Output: {y_onnx}")

# Load CORA and check
print(f"\n=== CORA Network ===")
nn = NeuralNetwork.readONNXNetwork(model_path)
x_cora = np.ones((4, 1))
y_cora = nn.evaluate(x_cora)
print(f"Input: {x_cora.flatten()}")
print(f"Output: {y_cora.flatten()}")

# Check first layer weights
if len(nn.layers) > 0 and hasattr(nn.layers[0], 'W'):
    print(f"\nFirst CORA layer:")
    print(f"  W shape: {nn.layers[0].W.shape}")
    print(f"  W:\n{nn.layers[0].W}")
    if hasattr(nn.layers[0], 'b'):
        print(f"  b: {nn.layers[0].b.flatten()}")

