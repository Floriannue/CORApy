"""
Test MatMul weight orientation
"""
import numpy as np
import onnxruntime as ort
import sys
sys.path.insert(0, 'cora_python')
from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.nn.neuralNetwork import NeuralNetwork
import os

model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'nn-nav-set.onnx')

# Load CORA network
nn = NeuralNetwork.readONNXNetwork(model_path)
W_cora = nn.layers[0].W
b_cora = nn.layers[0].b

print("=== CORA Layer 0 ===")
print(f"W shape: {W_cora.shape}")  # Should be [64, 4] for CORA
print(f"b shape: {b_cora.shape}")
print(f"W first row: {W_cora[0, :]}")

# Load ONNX weights directly
import onnx
model = onnx.load(model_path)
initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

W_onnx = initializers['fc_1_MatMul_W']
b_onnx = initializers['fc_1_Add_B']

print(f"\n=== ONNX Weights ===")
print(f"W shape: {W_onnx.shape}")  # [4, 64]
print(f"b shape: {b_onnx.shape}")
print(f"W first column (becomes first row after transpose): {W_onnx[:, 0]}")

# Test computation
x = np.ones((4, 1))
print(f"\n=== Test with x = ones(4, 1) ===")

# ONNX MatMul: Y = X * W where X is [1, 4], W is [4, 64]
x_onnx = x.T.astype(np.float32)  # [1, 4]
y_onnx_matmul = (x_onnx @ W_onnx).flatten()  # [1, 64]
y_onnx_full = (y_onnx_matmul + b_onnx)  # Add bias
print(f"ONNX MatMul (X @ W): first 5 = {y_onnx_matmul[:5]}")
print(f"ONNX Full (X @ W + b): first 5 = {y_onnx_full[:5]}")

# ONNX Runtime (full network)
sess = ort.InferenceSession(model_path)
y_onnx_runtime = sess.run(None, {sess.get_inputs()[0].name: x_onnx})[0]
print(f"ONNX Runtime (full network): {y_onnx_runtime.flatten()}")

# CORA: Y = W @ x + b where W is [64, 4], x is [4, 1]
y_cora_layer0 = (W_cora @ x + b_cora).flatten()
print(f"\nCORA Layer 0 (W @ x + b): first 5 = {y_cora_layer0[:5]}")

# CORA full network
y_cora_full = nn.evaluate(x)
print(f"CORA Full network: {y_cora_full.flatten()}")

# Check if weights match
print(f"\n=== Weight Orientation Check ===")
print(f"W_onnx shape: {W_onnx.shape} = [in_features, out_features] = [4, 64]")
print(f"W_cora shape: {W_cora.shape} = [out_features, in_features] = [64, 4]")
print(f"W_cora should equal W_onnx.T? {np.allclose(W_cora, W_onnx.T)}")
print(f"W_cora equals W_onnx.T? {np.allclose(W_cora, W_onnx.T)}")

if not np.allclose(W_cora, W_onnx.T):
    print(f"\nERROR: W_cora != W_onnx.T")
    print(f"Max difference: {np.max(np.abs(W_cora - W_onnx.T))}")
    print(f"\nW_cora[0, :] = {W_cora[0, :]}")
    print(f"W_onnx.T[0, :] = {W_onnx.T[0, :]}")
    print(f"W_onnx[:, 0] = {W_onnx[:, 0]}")

