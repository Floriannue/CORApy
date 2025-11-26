"""
Check all layers in CORA network
"""
import numpy as np
import sys
sys.path.insert(0, 'cora_python')
from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.nn.neuralNetwork import NeuralNetwork
import os
import onnx

model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'nn-nav-set.onnx')

# Load ONNX model
model = onnx.load(model_path)
initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

print("=== ONNX Weights ===")
print(f"fc_1_MatMul_W: {initializers['fc_1_MatMul_W'].shape}")
print(f"fc_1_Add_B: {initializers['fc_1_Add_B'].shape}")
print(f"fc_2_MatMul_W: {initializers['fc_2_MatMul_W'].shape}")
print(f"fc_2_Add_B: {initializers['fc_2_Add_B'].shape}")
print(f"fc_3_MatMul_W: {initializers['fc_3_MatMul_W'].shape}")
print(f"fc_3_Add_B: {initializers['fc_3_Add_B'].shape}")

# Load CORA network
nn = NeuralNetwork.readONNXNetwork(model_path)

print(f"\n=== CORA Network ===")
print(f"Total layers: {len(nn.layers)}")
for i, layer in enumerate(nn.layers):
    layer_type = type(layer).__name__
    print(f"\nLayer {i}: {layer_type}")
    if hasattr(layer, 'W'):
        print(f"  W shape: {layer.W.shape}")
        print(f"  W first row: {layer.W[0, :5] if layer.W.shape[1] >= 5 else layer.W[0, :]}")
    if hasattr(layer, 'b'):
        print(f"  b shape: {layer.b.shape}")
        print(f"  b first 5: {layer.b.flatten()[:5]}")
        print(f"  b all zeros? {np.allclose(layer.b, 0)}")

print(f"\n=== Expected Structure ===")
print("Layer 0: nnLinearLayer (fc_1) - W: (64, 4), b: (64, 1)")
print("Layer 1: nnReLULayer")
print("Layer 2: nnLinearLayer (fc_2) - W: (32, 64), b: (32, 1)")
print("Layer 3: nnReLULayer")
print("Layer 4: nnLinearLayer (fc_3) - W: (2, 32), b: (2, 1)")
print("Layer 5: nnTanhLayer")

