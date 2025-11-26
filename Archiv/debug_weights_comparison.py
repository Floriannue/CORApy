"""Debug script to compare weights from ONNX vs CORA"""
import onnx
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.nn.neuralNetwork.neuralNetwork import NeuralNetwork

# Find model path
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and not os.path.exists(os.path.join(current_dir, 'cora_matlab')):
    current_dir = os.path.dirname(current_dir)

# Load ONNX model
model_path = os.path.join(current_dir, 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')
model = onnx.load(model_path)
graph = model.graph

# Get Gemm node
gemm_node = None
for node in graph.node:
    if node.op_type == 'Gemm':
        gemm_node = node
        break

if gemm_node:
    print("ONNX Gemm Node:")
    print(f"  transB: {[attr.i for attr in gemm_node.attribute if attr.name == 'transB'][0] if any(attr.name == 'transB' for attr in gemm_node.attribute) else 0}")
    
    # Get weights and bias from ONNX
    initializers = {init.name: init for init in graph.initializer}
    weight_name = gemm_node.input[1]
    bias_name = gemm_node.input[2]
    
    onnx_weight = onnx.numpy_helper.to_array(initializers[weight_name])
    onnx_bias = onnx.numpy_helper.to_array(initializers[bias_name])
    
    print(f"\nONNX Weight shape: {onnx_weight.shape}")
    print(f"ONNX Weight (first 3x3):\n{onnx_weight[:3, :3]}")
    print(f"\nONNX Bias shape: {onnx_bias.shape}")
    print(f"ONNX Bias (first 5): {onnx_bias[:5]}")

# Load CORA network
nn = NeuralNetwork.readONNXNetwork(model_path, False, 'BCSS')

# Get the final linear layer (last layer with W and b)
final_layer = None
final_layer_idx = None
for i, layer in enumerate(reversed(nn.layers)):
    if hasattr(layer, 'W') and hasattr(layer, 'b'):
        final_layer = layer
        final_layer_idx = len(nn.layers) - 1 - i
        break

if final_layer:
    print(f"\n\nCORA Final Layer (index {final_layer_idx}):")
    print(f"  Layer type: {type(final_layer).__name__}")
    print(f"\n\nCORA Final Layer:")
    print(f"  Weight shape: {final_layer.W.shape}")
    print(f"  Weight (first 3x3):\n{final_layer.W[:3, :3]}")
    print(f"\n  Bias shape: {final_layer.b.shape}")
    print(f"  Bias (first 5): {final_layer.b[:5].flatten()}")
    
    # Compare
    print(f"\n\nComparison:")
    print(f"  Weight shapes match: {onnx_weight.shape == final_layer.W.shape}")
    if onnx_weight.shape == final_layer.W.shape:
        print(f"  Weight values match (first 3x3): {np.allclose(onnx_weight[:3, :3], final_layer.W[:3, :3])}")
        print(f"  Weight max diff: {np.max(np.abs(onnx_weight - final_layer.W))}")
    print(f"  Bias shapes match: {onnx_bias.shape == final_layer.b.shape or (onnx_bias.size == final_layer.b.size)}")
    if onnx_bias.size == final_layer.b.size:
        print(f"  Bias values match (first 5): {np.allclose(onnx_bias[:5], final_layer.b[:5].flatten())}")
        print(f"  Bias max diff: {np.max(np.abs(onnx_bias - final_layer.b.flatten()))}")
