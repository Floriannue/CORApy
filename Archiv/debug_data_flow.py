"""Debug script to trace data flow through the network"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.nn.neuralNetwork.neuralNetwork import NeuralNetwork
import onnxruntime as ort

# Find model path
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and not os.path.exists(os.path.join(current_dir, 'cora_matlab')):
    current_dir = os.path.dirname(current_dir)

model_path = os.path.join(current_dir, 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')

# Load CORA network
nn = NeuralNetwork.readONNXNetwork(model_path, False, 'BCSS')

# Create test input
x = np.ones((nn.neurons_in, 1))
print(f"Input shape: {x.shape}")
print(f"Input (first 10): {x[:10].flatten()}\n")

# Trace through each layer
current = x
for i, layer in enumerate(nn.layers):
    print(f"Layer {i}: {type(layer).__name__}")
    if hasattr(layer, 'inputSize'):
        print(f"  inputSize: {layer.inputSize}")
    
    # Evaluate layer
    current = layer.evaluateNumeric(current, {})
    print(f"  Output shape: {current.shape}")
    print(f"  Output (first 10): {current[:10].flatten() if current.size >= 10 else current.flatten()}")
    
    # Special check for Reshape layer
    if type(layer).__name__ == 'nnReshapeLayer':
        print(f"  Reshape idx_out: {layer.idx_out}")
    
    # Special check for AvgPool layer
    if type(layer).__name__ == 'nnAvgPool2DLayer':
        print(f"  AvgPool W shape: {layer.W.shape}")
        print(f"  AvgPool W[:, :, 0, 0] (first channel): {layer.W[:, :, 0, 0]}")
        print(f"  AvgPool W[:, :, 1, 1] (second channel): {layer.W[:, :, 1, 1]}")
        print(f"  AvgPool W[:, :, 0, 1] (cross channel, should be 0): {layer.W[:, :, 0, 1]}")
        print(f"  AvgPool poolSize: {layer.poolSize}")
        print(f"  AvgPool stride: {layer.stride}")
    
    print()

print(f"\nFinal CORA output shape: {current.shape}")
print(f"Final CORA output: {current.flatten()}")

# Compare with ONNX Runtime
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Prepare input for ONNX (BCSS format: [B, C, H, W])
x_onnx = x.reshape(28, 28, 1).transpose(2, 0, 1).reshape(1, 1, 28, 28).astype(np.float32)
outputs = session.run(None, {input_name: x_onnx})
y_onnx = outputs[0]

print(f"\nONNX Runtime output shape: {y_onnx.shape}")
print(f"ONNX Runtime output: {y_onnx.flatten()}")

print(f"\nDifference: {np.abs(current.flatten() - y_onnx.flatten())}")

