"""
Debug script to test convolutional network input
"""
import numpy as np
import sys
import os
sys.path.insert(0, 'cora_python')

from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.nn.neuralNetwork import NeuralNetwork

model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')

print(f"Loading model: {model_path}")
nn = NeuralNetwork.readONNXNetwork(model_path, False, 'BCSS')

print(f"\nNetwork info:")
print(f"  neurons_in: {nn.neurons_in}")
print(f"  neurons_out: {nn.neurons_out}")
print(f"  Number of layers: {len(nn.layers)}")

print(f"\nFirst layer:")
layer0 = nn.layers[0]
print(f"  Type: {type(layer0).__name__}")
print(f"  inputSize: {layer0.inputSize if hasattr(layer0, 'inputSize') else 'N/A'}")

if hasattr(layer0, 'W'):
    print(f"  W shape: {layer0.W.shape}")
if hasattr(layer0, 'poolSize'):
    print(f"  poolSize: {layer0.poolSize}")
if hasattr(layer0, 'stride'):
    print(f"  stride: {layer0.stride}")

print(f"\nCreating test input:")
x = np.ones((nn.neurons_in, 1))
print(f"  x shape: {x.shape}")
print(f"  x size: {x.size}")

print(f"\nExpected input size for first layer:")
if hasattr(layer0, 'inputSize') and layer0.inputSize:
    expected_size = np.prod(layer0.inputSize)
    print(f"  inputSize: {layer0.inputSize}")
    print(f"  Expected flattened size: {expected_size}")
    print(f"  Actual x size: {x.size}")
    print(f"  Match: {expected_size == x.size}")

print(f"\nAttempting to evaluate...")
try:
    y = nn.evaluate(x)
    print(f"Success! Output shape: {y.shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

