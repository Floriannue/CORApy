"""
Debug script to compare CORA vs ONNX Runtime for convolutional network
Matches MATLAB debug script structure
"""
import numpy as np
import os
import onnxruntime as ort
import sys
sys.path.insert(0, 'cora_python')

from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.nn.neuralNetwork import NeuralNetwork

# Test convolutional network
model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}\n")

# Load CORA network
nn = NeuralNetwork.readONNXNetwork(model_path, False, 'BCSS')
print(f"CORA Network:")
print(f"  neurons_in: {nn.neurons_in}")
print(f"  neurons_out: {nn.neurons_out}")
print(f"  Number of layers: {len(nn.layers)}")
for i, layer in enumerate(nn.layers):
    print(f"    Layer {i}: {type(layer).__name__}")
    if hasattr(layer, 'inputSize') and layer.inputSize:
        print(f"      inputSize: {layer.inputSize}")
print()

# Test input
x = np.ones((nn.neurons_in, 1))
print(f"Test input:")
print(f"  x shape: {x.shape}")
print(f"  x (first 10): {x[:10].flatten()}")
print()

# CORA evaluation
y_cora = nn.evaluate(x)
print(f"CORA output:")
print(f"  y_cora shape: {y_cora.shape}")
print(f"  y_cora: {y_cora.flatten()}")
print()

# Get input shape from first layer
input_shape = None
if hasattr(nn.layers[0], 'inputSize') and nn.layers[0].inputSize:
    input_shape = nn.layers[0].inputSize
    print(f"Input shape from first layer: {input_shape}")
    print()

# ONNX Runtime evaluation
sess = ort.InferenceSession(model_path)
onnx_input = sess.get_inputs()[0]
print(f"ONNX Model Input:")
print(f"  Name: {onnx_input.name}")
print(f"  Shape: {onnx_input.shape}")
print(f"  Type: {onnx_input.type}")
print()

# Prepare input for ONNX (matching MATLAB: reshape(x, inputSize))
if input_shape is not None:
    x_flat = x.flatten()
    print(f"Reshaping input:")
    print(f"  x_flat shape: {x_flat.shape}")
    
    if len(input_shape) == 3:
        H, W, C = input_shape
        x_reshaped = x_flat.reshape(H, W, C)
        print(f"  Reshaped to [H, W, C]: {x_reshaped.shape}")
        
        # Check ONNX format
        onnx_dims = []
        for dim in onnx_input.shape:
            if isinstance(dim, int) and dim > 0:
                onnx_dims.append(dim)
            else:
                onnx_dims.append(None)
        
        print(f"  ONNX expected shape: {onnx_input.shape}")
        print(f"  ONNX dims (processed): {onnx_dims}")
        
        # Try BCSS format: [B, C, H, W]
        if onnx_dims[1] == C and (onnx_dims[2] == H or onnx_dims[2] is None):
            print(f"  Using BCSS format: [B, C, H, W]")
            x_onnx = x_reshaped.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
            x_onnx = np.expand_dims(x_onnx, axis=0)  # [C, H, W] -> [1, C, H, W]
        else:
            print(f"  Using BSSC format: [B, H, W, C]")
            x_onnx = np.expand_dims(x_reshaped, axis=0)  # [H, W, C] -> [1, H, W, C]
        
        print(f"  Final ONNX input shape: {x_onnx.shape}")
        print(f"  Final ONNX input (first 20): {x_onnx.flatten()[:20]}")
        print()
    else:
        x_onnx = x_flat.reshape(1, -1)
        print(f"  Reshaped to [1, features]: {x_onnx.shape}")
        print()
else:
    x_onnx = x.T
    print(f"Using transpose (feed-forward): {x_onnx.shape}")
    print()

# Run ONNX inference
x_onnx = x_onnx.astype(np.float32)
y_onnx = sess.run(None, {onnx_input.name: x_onnx})[0]

# Convert to column vector
if y_onnx.ndim == 1:
    y_onnx = y_onnx.reshape(-1, 1)
else:
    y_onnx = y_onnx[0].reshape(-1, 1)

print(f"ONNX Runtime output:")
print(f"  y_onnx shape: {y_onnx.shape}")
print(f"  y_onnx: {y_onnx.flatten()}")
print()

# Compare
tol = 1e-6
diff = np.abs(y_cora - y_onnx)
print(f"Comparison:")
print(f"  Max difference: {np.max(diff):.6e}")
print(f"  Mean difference: {np.mean(diff):.6e}")
print(f"  Differences: {diff.flatten()}")
print(f"  Within tolerance (1e-6): {np.all(np.abs(diff) < tol)}")
print()

# Trace through CORA layers step by step
print("Tracing through CORA layers:")
current = x.copy()
for i, layer in enumerate(nn.layers):
    print(f"\nLayer {i}: {type(layer).__name__}")
    print(f"  Input shape: {current.shape}")
    print(f"  Input (first 10): {current[:10].flatten() if current.size >= 10 else current.flatten()}")
    
    current = layer.evaluateNumeric(current, {})
    print(f"  Output shape: {current.shape}")
    print(f"  Output (first 10): {current[:10].flatten() if current.size >= 10 else current.flatten()}")
    
    # Special info for specific layers
    if type(layer).__name__ == 'nnConv2DLayer':
        if hasattr(layer, 'W') and layer.W is not None:
            print(f"  Conv W shape: {layer.W.shape}")
            print(f"  Conv W (first 5x5): {layer.W[:5, :5, 0, 0] if layer.W.ndim == 4 else 'N/A'}")
        if hasattr(layer, 'b') and layer.b is not None:
            print(f"  Conv b shape: {layer.b.shape}")
            print(f"  Conv b: {layer.b.flatten()[:5]}")
    elif type(layer).__name__ == 'nnAvgPool2DLayer':
        if hasattr(layer, 'W') and layer.W is not None:
            print(f"  AvgPool W shape: {layer.W.shape}")
    elif type(layer).__name__ == 'nnLinearLayer':
        if hasattr(layer, 'W') and layer.W is not None:
            print(f"  Linear W shape: {layer.W.shape}")
            print(f"  Linear W (first 5x5): {layer.W[:5, :5] if layer.W.ndim == 2 else 'N/A'}")
        if hasattr(layer, 'b') and layer.b is not None:
            print(f"  Linear b shape: {layer.b.shape}")
            print(f"  Linear b: {layer.b.flatten()[:5]}")

print(f"\nFinal CORA output: {current.flatten()}")

