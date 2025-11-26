"""
Analyze the ONNX to CORA conversion issue
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

# Get ONNX weights
initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

print("=== ONNX Model Structure ===")
gemm_nodes = [n for n in model.graph.node if n.op_type == 'Gemm']
print(f"Found {len(gemm_nodes)} Gemm nodes")

if gemm_nodes:
    node = gemm_nodes[0]
    print(f"\nFirst Gemm node: {node.name}")
    print(f"  Inputs: {list(node.input)}")
    
    # Get attributes
    attrs = {}
    for attr in node.attribute:
        attrs[attr.name] = onnx.helper.get_attribute_value(attr)
    print(f"  Attributes: {attrs}")
    print(f"  Default transB=1 if not specified")
    
    # Get weight
    if len(node.input) >= 2:
        w_name = node.input[1]
        if w_name in initializers:
            W_onnx = initializers[w_name]
            print(f"\n  ONNX Weight '{w_name}':")
            print(f"    Shape: {W_onnx.shape}")
            print(f"    First row: {W_onnx[0, :]}")
            print(f"    First column: {W_onnx[:, 0]}")
    
    # Get bias
    if len(node.input) >= 3:
        b_name = node.input[2]
        if b_name in initializers:
            b_onnx = initializers[b_name]
            print(f"  ONNX Bias '{b_name}':")
            print(f"    Shape: {b_onnx.shape}")
            print(f"    First 5 values: {b_onnx[:5]}")

# Load CORA network
print(f"\n=== CORA Network ===")
nn = NeuralNetwork.readONNXNetwork(model_path)
print(f"Number of layers: {len(nn.layers)}")

if len(nn.layers) > 0 and hasattr(nn.layers[0], 'W'):
    W_cora = nn.layers[0].W
    b_cora = nn.layers[0].b
    print(f"\nCORA first layer:")
    print(f"  W shape: {W_cora.shape}")
    print(f"  W first row: {W_cora[0, :]}")
    print(f"  W first column: {W_cora[:, 0]}")
    print(f"  b shape: {b_cora.shape}")
    print(f"  b first 5 values: {b_cora[:5].flatten()}")

# Compare weights
print(f"\n=== Weight Comparison ===")
if 'W_onnx' in locals() and 'W_cora' in locals():
    print(f"ONNX W shape: {W_onnx.shape}")
    print(f"CORA W shape: {W_cora.shape}")
    print(f"\nDirect comparison:")
    print(f"  W_onnx == W_cora? {np.allclose(W_onnx, W_cora)}")
    print(f"  W_onnx == W_cora.T? {np.allclose(W_onnx, W_cora.T)}")
    print(f"  W_onnx.T == W_cora? {np.allclose(W_onnx.T, W_cora)}")

# Test forward pass manually
print(f"\n=== Manual Forward Pass ===")
x = np.ones((4, 1))
print(f"Input x: {x.flatten()}")

# ONNX Runtime
sess = ort.InferenceSession(model_path)
x_onnx = x.T.astype(np.float32)  # [1, 4]
y_onnx = sess.run(None, {sess.get_inputs()[0].name: x_onnx})[0]
print(f"\nONNX Runtime:")
print(f"  Input shape: {x_onnx.shape}")
print(f"  Output: {y_onnx.flatten()}")

# CORA
y_cora = nn.evaluate(x)
print(f"\nCORA:")
print(f"  Input shape: {x.shape}")
print(f"  Output: {y_cora.flatten()}")

# Manual computation with ONNX weights
if 'W_onnx' in locals() and 'b_onnx' in locals():
    print(f"\n=== Manual Computation with ONNX Weights ===")
    
    # ONNX Gemm with default transB=1: Y = X * W^T + b
    # X is [1, 4], W is [64, 4], W^T is [4, 64]
    # Y = [1, 4] * [4, 64] + b = [1, 64]
    y_manual_onnx = (x_onnx @ W_onnx.T + b_onnx).flatten()
    print(f"Manual ONNX (X @ W^T + b): {y_manual_onnx[:5]}... (first 5)")
    print(f"  Matches ONNX Runtime? {np.allclose(y_manual_onnx, y_onnx.flatten())}")
    
    # CORA: Y = W @ x + b
    # W is [64, 4], x is [4, 1]
    # Y = [64, 4] * [4, 1] + b = [64, 1]
    y_manual_cora = (W_cora @ x + b_cora).flatten()
    print(f"\nManual CORA (W @ x + b): {y_manual_cora[:5]}... (first 5)")
    print(f"  Matches CORA? {np.allclose(y_manual_cora, y_cora.flatten())}")
    
    # What should W_cora be to match ONNX?
    # ONNX: Y = X * W_onnx^T + b, where X is [1, 4], W_onnx is [64, 4]
    # CORA: Y = W_cora @ x + b, where x is [4, 1], W_cora is [64, 4]
    # For same result: X * W_onnx^T = W_cora @ x
    # [1, 4] * [4, 64] = [64, 4] * [4, 1]
    # Both give [1, 64] or [64, 1]
    # So W_cora should equal W_onnx (no transpose!)
    print(f"\n=== Conclusion ===")
    print(f"For ONNX Gemm with transB=1 (default):")
    print(f"  ONNX computes: Y = X * W^T + b where X is [batch, in], W is [out, in]")
    print(f"  CORA computes: Y = W @ x + b where x is [in, 1], W is [out, in]")
    print(f"  Therefore: W_cora should equal W_onnx (NO TRANSPOSE)")
    print(f"\nCurrent code does W.T, which is WRONG!")

