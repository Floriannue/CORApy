"""Debug the first window of AvgPool to see what's happening"""
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

# Create test input
x = np.ones((nn.neurons_in, 1))

# Trace through layers
current = x
for i, layer in enumerate(nn.layers):
    if i == 2:  # AvgPool layer
        print(f"Layer {i}: {type(layer).__name__}")
        print(f"  inputSize: {layer.inputSize}")
        print(f"  W shape: {layer.W.shape}")
        print(f"  W[:, :, 0, 0]:\n{layer.W[:, :, 0, 0]}")
        print(f"  stride: {layer.stride}")
        print(f"  poolSize: {layer.poolSize}")
        
        # Check input to this layer
        print(f"\n  Input to AvgPool (first 20): {current[:20].flatten()}")
        
        # Manually compute what the first output should be
        # Input is [27, 27, 32] flattened to (23328, 1)
        # Reshape to [27, 27, 32, 1]
        input_reshaped = current.reshape(27, 27, 32, 1)
        # First channel, first 4x4 window
        first_window = input_reshaped[0:4, 0:4, 0, 0]
        print(f"  First 4x4 window of channel 0:\n{first_window}")
        print(f"  Sum of first window: {np.sum(first_window)}")
        print(f"  Average (sum/16): {np.sum(first_window) / 16}")
        
        # What the convolution should produce
        filter_channel = layer.W[:, :, 0, 0]
        conv_result_manual = np.sum(first_window * filter_channel)
        print(f"  Manual convolution result (window * filter): {conv_result_manual}")
        
        # Evaluate layer
        output = layer.evaluateNumeric(current, {})
        print(f"  Output from layer (first 20): {output[:20].flatten()}")
        print(f"  First output value: {output[0, 0]}")
        print(f"  Expected first output: {conv_result_manual}")
        print(f"  Match? {np.isclose(output[0, 0], conv_result_manual)}")
        
        current = output
        break
    else:
        current = layer.evaluateNumeric(current, {})

