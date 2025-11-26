"""
Debug weight orientation and Gemm attributes
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

print("=== ONNX Model Analysis ===")
initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

# Find first Gemm node
for i, node in enumerate(model.graph.node):
    if node.op_type == 'Gemm':
        print(f"\nGemm Node {i}: {node.name}")
        print(f"  Inputs: {node.input}")
        
        # Get attributes
        transA = 0
        transB = 1  # Default
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
                print(f"    First row: {W_onnx[0, :]}")
                
                # Gemm computes: Y = alpha * (A^transA) * (B^transB) + beta * C
                # With transB=1 (default): Y = A * B^T
                # So if B is [out, in], we compute A * B^T = [batch, in] * [in, out] = [batch, out]
                if transB:
                    print(f"    With transB=1, effective weight is B^T")
                    print(f"    B^T shape would be: {W_onnx.T.shape}")
                    print(f"    B^T first row: {W_onnx.T[0, :]}")
                else:
                    print(f"    With transB=0, effective weight is B")
        
        if len(node.input) >= 3:
            bias_name = node.input[2]
            if bias_name in initializers:
                b_onnx = initializers[bias_name]
                print(f"  ONNX Bias '{bias_name}':")
                print(f"    Shape: {b_onnx.shape}")
                print(f"    Values: {b_onnx.flatten()}")
        
        break  # Only check first Gemm

# Load CORA network
print(f"\n=== CORA Network Analysis ===")
nn = NeuralNetwork.readONNXNetwork(model_path)

if len(nn.layers) > 0 and hasattr(nn.layers[0], 'W'):
    W_cora = nn.layers[0].W
    print(f"CORA first layer:")
    print(f"  W shape: {W_cora.shape}")
    print(f"  W first row: {W_cora[0, :]}")
    if hasattr(nn.layers[0], 'b'):
        b_cora = nn.layers[0].b
        print(f"  b shape: {b_cora.shape}")
        print(f"  b values: {b_cora.flatten()}")

# Test computation manually
print(f"\n=== Manual Computation Test ===")
x = np.ones((1, 4), dtype=np.float32)  # [batch=1, features=4]

# ONNX Runtime
sess = ort.InferenceSession(model_path)
y_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0]
print(f"ONNX Runtime output: {y_onnx.flatten()}")

# CORA
x_cora = np.ones((4, 1))  # [features=4, 1]
y_cora = nn.evaluate(x_cora)
print(f"CORA output: {y_cora.flatten()}")

# Manual computation with ONNX weights
if len(nn.layers) > 0 and hasattr(nn.layers[0], 'W'):
    W_onnx = initializers[weight_name]
    b_onnx = initializers[bias_name] if len(node.input) >= 3 and node.input[2] in initializers else np.zeros(W_onnx.shape[0])
    
    # Gemm with transB=1: Y = X * W^T + b
    # X is [1, 4], W is [out, in], so W^T is [in, out] = [4, out]
    # Y = [1, 4] * [4, out] + b = [1, out]
    if transB:
        W_effective = W_onnx.T  # [in, out]
    else:
        W_effective = W_onnx  # [out, in]
    
    y_manual = x @ W_effective + b_onnx.flatten()
    print(f"Manual computation (X @ W_effective + b): {y_manual.flatten()}")
    print(f"  W_effective shape: {W_effective.shape}")
    print(f"  W_effective first row: {W_effective[0, :] if W_effective.shape[0] > 0 else 'N/A'}")

