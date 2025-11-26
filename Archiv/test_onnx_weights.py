"""
Test ONNX weight extraction and conversion
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

print("=== ONNX Gemm Analysis ===")
initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

# Find first Gemm node
for i, node in enumerate(model.graph.node):
    if node.op_type == 'Gemm':
        print(f"\nGemm Node {i}: {node.name}")
        
        # Get attributes (defaults: transA=0, transB=1, alpha=1.0, beta=1.0)
        transA = 0
        transB = 1
        alpha = 1.0
        beta = 1.0
        
        for attr in node.attribute:
            val = onnx.helper.get_attribute_value(attr)
            if attr.name == 'transA':
                transA = int(val)
            elif attr.name == 'transB':
                transB = int(val)
            elif attr.name == 'alpha':
                alpha = float(val)
            elif attr.name == 'beta':
                beta = float(val)
            print(f"  {attr.name}: {val}")
        
        # Get weight and bias
        if len(node.input) >= 2:
            weight_name = node.input[1]
            if weight_name in initializers:
                W_onnx = initializers[weight_name]
                print(f"\n  ONNX Weight '{weight_name}':")
                print(f"    Shape: {W_onnx.shape}")
                print(f"    First row: {W_onnx[0, :5] if W_onnx.shape[1] >= 5 else W_onnx[0, :]}")
        
        if len(node.input) >= 3:
            bias_name = node.input[2]
            if bias_name in initializers:
                b_onnx = initializers[bias_name]
                print(f"  ONNX Bias '{bias_name}':")
                print(f"    Shape: {b_onnx.shape}")
                print(f"    First 5 values: {b_onnx.flatten()[:5]}")
        
        break

# Load CORA network
print(f"\n=== CORA Network ===")
nn = NeuralNetwork.readONNXNetwork(model_path)

if len(nn.layers) > 0 and hasattr(nn.layers[0], 'W'):
    W_cora = nn.layers[0].W
    print(f"CORA first layer W:")
    print(f"  Shape: {W_cora.shape}")
    print(f"  First row: {W_cora[0, :]}")
    
    # Compare with ONNX
    if 'weight_name' in locals() and weight_name in initializers:
        W_onnx = initializers[weight_name]
        print(f"\n=== Comparison ===")
        print(f"ONNX W shape: {W_onnx.shape}")
        print(f"CORA W shape: {W_cora.shape}")
        print(f"Are they equal? {np.allclose(W_onnx, W_cora, atol=1e-6)}")
        print(f"Are ONNX and CORA.T equal? {np.allclose(W_onnx, W_cora.T, atol=1e-6)}")
        print(f"Are ONNX.T and CORA equal? {np.allclose(W_onnx.T, W_cora, atol=1e-6)}")
        
        if transB:
            print(f"\nWith transB=1, ONNX computes: Y = X * W_onnx^T")
            print(f"  W_onnx^T shape: {W_onnx.T.shape}")
            print(f"  CORA computes: Y = W_cora @ X")
            print(f"  For these to match, we need W_cora = W_onnx^T")
            print(f"  W_cora == W_onnx^T? {np.allclose(W_cora, W_onnx.T, atol=1e-6)}")

# Test computation
print(f"\n=== Computation Test ===")
x_onnx = np.ones((1, 4), dtype=np.float32)  # [batch, features]
sess = ort.InferenceSession(model_path)
y_onnx = sess.run(None, {sess.get_inputs()[0].name: x_onnx})[0]
print(f"ONNX input: {x_onnx}")
print(f"ONNX output: {y_onnx.flatten()}")

x_cora = np.ones((4, 1))  # [features, 1]
y_cora = nn.evaluate(x_cora)
print(f"CORA input: {x_cora.flatten()}")
print(f"CORA output: {y_cora.flatten()}")

