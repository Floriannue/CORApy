"""
Debug script to compare CORA vs ONNX Runtime evaluation
"""
import numpy as np
import os
import onnxruntime as ort
import sys
sys.path.insert(0, 'cora_python')

from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.nn.neuralNetwork import NeuralNetwork

# Test feed-forward network
model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'nn-nav-set.onnx')
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

# Load CORA network
nn = NeuralNetwork.readONNXNetwork(model_path)
print(f"\nCORA Network:")
print(f"  neurons_in: {nn.neurons_in}")
print(f"  neurons_out: {nn.neurons_out}")
print(f"  Number of layers: {len(nn.layers)}")

# Check ONNX model input/output
sess = ort.InferenceSession(model_path)
onnx_input = sess.get_inputs()[0]
onnx_output = sess.get_outputs()[0]
print(f"\nONNX Model:")
print(f"  Input name: {onnx_input.name}")
print(f"  Input shape: {onnx_input.shape}")
print(f"  Input type: {onnx_input.type}")
print(f"  Output name: {onnx_output.name}")
print(f"  Output shape: {onnx_output.shape}")
print(f"  Output type: {onnx_output.type}")

# Test input
x = np.ones((nn.neurons_in, 1))
print(f"\nTest input:")
print(f"  x shape: {x.shape}")
print(f"  x values: {x.flatten()}")

# CORA evaluation
y_cora = nn.evaluate(x)
print(f"\nCORA output:")
print(f"  y_cora shape: {y_cora.shape}")
print(f"  y_cora values: {y_cora.flatten()}")

# ONNX Runtime evaluation - try different input formats
print(f"\nONNX Runtime evaluation:")

# Format 1: x.T (row vector)
x1 = x.T.astype(np.float32)
print(f"  Format 1 (x.T): shape={x1.shape}, values={x1.flatten()}")
y1 = sess.run(None, {onnx_input.name: x1})[0]
print(f"    Output shape: {y1.shape}, values: {y1.flatten()}")
print(f"    As column: {y1[0].reshape(-1, 1).flatten()}")

# Format 2: x.flatten() then reshape
x2 = x.flatten().reshape(1, -1).astype(np.float32)
print(f"  Format 2 (flatten->reshape): shape={x2.shape}, values={x2.flatten()}")
y2 = sess.run(None, {onnx_input.name: x2})[0]
print(f"    Output shape: {y2.shape}, values: {y2.flatten()}")
print(f"    As column: {y2[0].reshape(-1, 1).flatten()}")

# Format 3: Direct reshape
x3 = x.reshape(1, -1).astype(np.float32)
print(f"  Format 3 (reshape): shape={x3.shape}, values={x3.flatten()}")
y3 = sess.run(None, {onnx_input.name: x3})[0]
print(f"    Output shape: {y3.shape}, values: {y3.flatten()}")
print(f"    As column: {y3[0].reshape(-1, 1).flatten()}")

# Check if any match CORA
print(f"\nComparison:")
print(f"  CORA: {y_cora.flatten()}")
print(f"  ONNX Format 1: {y1[0].reshape(-1, 1).flatten()}")
print(f"  Match Format 1: {np.allclose(y_cora.flatten(), y1[0].reshape(-1, 1).flatten(), atol=1e-6)}")
print(f"  ONNX Format 2: {y2[0].reshape(-1, 1).flatten()}")
print(f"  Match Format 2: {np.allclose(y_cora.flatten(), y2[0].reshape(-1, 1).flatten(), atol=1e-6)}")
print(f"  ONNX Format 3: {y3[0].reshape(-1, 1).flatten()}")
print(f"  Match Format 3: {np.allclose(y_cora.flatten(), y3[0].reshape(-1, 1).flatten(), atol=1e-6)}")

