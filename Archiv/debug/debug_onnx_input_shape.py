"""
Debug script to check ONNX input shape requirements
"""
import numpy as np
import os
import onnxruntime as ort
import sys
sys.path.insert(0, 'cora_python')

from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.nn.neuralNetwork import NeuralNetwork

# Test VNN-COMP network
model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx')
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}\n")

# Load CORA network
nn = NeuralNetwork.readONNXNetwork(model_path, False, 'BSSC')
print(f"CORA Network:")
print(f"  neurons_in: {nn.neurons_in}")
print(f"  neurons_out: {nn.neurons_out}")
print(f"  Number of layers: {len(nn.layers)}")
print(f"  First layer type: {type(nn.layers[0]).__name__}")
if hasattr(nn.layers[0], 'inputSize') and nn.layers[0].inputSize:
    print(f"  First layer inputSize: {nn.layers[0].inputSize}")
print()

# Check ONNX model input
sess = ort.InferenceSession(model_path)
onnx_input = sess.get_inputs()[0]
print(f"ONNX Model Input:")
print(f"  Name: {onnx_input.name}")
print(f"  Shape: {onnx_input.shape}")
print(f"  Type: {onnx_input.type}")
print()

# Generate test input
x = np.random.rand(nn.neurons_in, 1)
print(f"Test input:")
print(f"  x shape: {x.shape}")
print(f"  x (first 5): {x[:5].flatten()}")
print()

# Try different input formats
print("Trying different input formats:")
print()

# Format 1: [1, features] - feed-forward
x1 = x.T.astype(np.float32)
print(f"Format 1 (x.T): shape={x1.shape}")
try:
    y1 = sess.run(None, {onnx_input.name: x1})[0]
    print(f"  SUCCESS - Output shape: {y1.shape}")
except Exception as e:
    print(f"  FAILED - {e}")

# Format 2: Reshape based on inputSize if available
if hasattr(nn.layers[0], 'inputSize') and nn.layers[0].inputSize:
    input_shape = nn.layers[0].inputSize
    print(f"\nFormat 2 (using inputSize {input_shape}):")
    
    if len(input_shape) == 3:
        H, W, C = input_shape
        x2_flat = x.flatten()
        x2 = x2_flat.reshape(H, W, C)
        # BSSC: [B, H, W, C]
        x2 = np.expand_dims(x2, axis=0).astype(np.float32)
        print(f"  Reshaped to: {x2.shape}")
        try:
            y2 = sess.run(None, {onnx_input.name: x2})[0]
            print(f"  SUCCESS - Output shape: {y2.shape}")
        except Exception as e:
            print(f"  FAILED - {e}")
    elif len(input_shape) == 1:
        x2 = x.flatten().reshape(1, -1).astype(np.float32)
        print(f"  Reshaped to: {x2.shape}")
        try:
            y2 = sess.run(None, {onnx_input.name: x2})[0]
            print(f"  SUCCESS - Output shape: {y2.shape}")
        except Exception as e:
            print(f"  FAILED - {e}")

# Format 3: Try to match ONNX input shape exactly
print(f"\nFormat 3 (match ONNX shape {onnx_input.shape}):")
# Handle dynamic dimensions
expected_shape = []
for dim in onnx_input.shape:
    if isinstance(dim, int) and dim > 0:
        expected_shape.append(dim)
    else:
        expected_shape.append(1)  # Use 1 for dynamic dimensions

# Calculate total elements needed
total_elements = np.prod(expected_shape)
if total_elements == x.size:
    x3 = x.flatten().reshape(expected_shape).astype(np.float32)
    print(f"  Reshaped to: {x3.shape}")
    try:
        y3 = sess.run(None, {onnx_input.name: x3})[0]
        print(f"  SUCCESS - Output shape: {y3.shape}")
    except Exception as e:
        print(f"  FAILED - {e}")
else:
    print(f"  Cannot reshape: x has {x.size} elements, but ONNX expects {total_elements}")

