"""
Debug script to inspect ONNX model structure and compare with CORA conversion
"""
import numpy as np
import os
import onnx
import sys
sys.path.insert(0, 'cora_python')

from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.nn.neuralNetwork import NeuralNetwork

# Load ONNX model
model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'nn-nav-set.onnx')
model = onnx.load(model_path)

print("=== ONNX Model Structure ===")
print(f"Inputs: {[inp.name for inp in model.graph.input]}")
print(f"Outputs: {[out.name for out in model.graph.output]}")

# Get initializers (weights/biases)
initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}
print(f"\nInitializers (weights/biases):")
for name, arr in initializers.items():
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

# Check graph nodes
print(f"\nGraph nodes ({len(model.graph.node)}):")
for i, node in enumerate(model.graph.node):
    print(f"  Node {i}: {node.op_type}")
    print(f"    Inputs: {list(node.input)}")
    print(f"    Outputs: {list(node.output)}")
    if node.op_type == 'Gemm':
        # Check weights
        for inp in node.input:
            if inp in initializers:
                w = initializers[inp]
                print(f"    Weight {inp}: shape={w.shape}")

# Load CORA network and check weights
print(f"\n=== CORA Network Structure ===")
nn = NeuralNetwork.readONNXNetwork(model_path)
print(f"Layers: {len(nn.layers)}")
for i, layer in enumerate(nn.layers):
    print(f"  Layer {i}: {type(layer).__name__}")
    if hasattr(layer, 'W'):
        print(f"    W shape: {layer.W.shape}")
        print(f"    W sample (first row): {layer.W[0, :5] if layer.W.shape[1] >= 5 else layer.W[0, :]}")
    if hasattr(layer, 'b'):
        print(f"    b shape: {layer.b.shape}")
        print(f"    b values: {layer.b.flatten()}")

# Compare first layer weights with ONNX
print(f"\n=== Weight Comparison ===")
if len(nn.layers) > 0 and hasattr(nn.layers[0], 'W'):
    cora_W = nn.layers[0].W
    print(f"CORA first layer W shape: {cora_W.shape}")
    print(f"CORA first layer W sample: {cora_W[0, :]}")
    
    # Find corresponding ONNX weight
    for node in model.graph.node:
        if node.op_type == 'Gemm':
            for inp in node.input:
                if inp in initializers and initializers[inp].ndim == 2:
                    onnx_W = initializers[inp]
                    print(f"ONNX Gemm weight shape: {onnx_W.shape}")
                    print(f"ONNX Gemm weight sample: {onnx_W[0, :]}")
                    print(f"Are they the same? {np.allclose(cora_W, onnx_W.T, atol=1e-6)}")
                    if not np.allclose(cora_W, onnx_W.T, atol=1e-6):
                        print(f"  Difference: max={np.max(np.abs(cora_W - onnx_W.T))}")
                    break

