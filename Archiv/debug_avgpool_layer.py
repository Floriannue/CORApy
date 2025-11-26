"""Debug script to check AvgPool layer parameters"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.nn.neuralNetwork.neuralNetwork import NeuralNetwork

# Find model path
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and not os.path.exists(os.path.join(current_dir, 'cora_matlab')):
    current_dir = os.path.dirname(current_dir)

model_path = os.path.join(current_dir, 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')

# Load CORA network
nn = NeuralNetwork.readONNXNetwork(model_path, False, 'BCSS')

# Find AvgPool layer
avgpool_layer = None
for i, layer in enumerate(nn.layers):
    if hasattr(layer, 'poolSize'):
        avgpool_layer = layer
        print(f"Found AvgPool layer at index {i}")
        print(f"  Type: {type(layer).__name__}")
        print(f"  poolSize: {layer.poolSize}")
        print(f"  stride: {layer.stride}")
        print(f"  padding: {layer.padding}")
        print(f"  dilation: {layer.dilation}")
        print(f"  inputSize: {layer.inputSize}")
        print(f"  W shape: {layer.W.shape}")
        print(f"  b shape: {layer.b.shape}")
        print(f"  W (first 2x2x2x2):\n{layer.W[:2, :2, :2, :2]}")
        print(f"  b (first 5): {layer.b[:5]}")
        break

