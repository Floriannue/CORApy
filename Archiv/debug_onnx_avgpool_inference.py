"""Debug script to infer AvgPool parameters from ONNX Runtime"""
import onnxruntime as ort
import numpy as np
import os

# Find model path
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and not os.path.exists(os.path.join(current_dir, 'cora_matlab')):
    current_dir = os.path.dirname(current_dir)

model_path = os.path.join(current_dir, 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')

session = ort.InferenceSession(model_path)

# Get input/output info
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape

print(f"Input: {input_name}, shape: {input_shape}")
print(f"Output: {output_name}, shape: {output_shape}")

# Create test input: all ones
x = np.ones((1, 1, 28, 28), dtype=np.float32)

# Run through each layer to see intermediate outputs
# We need to manually trace through the graph
import onnx
model = onnx.load(model_path)
graph = model.graph

print("\nTracing through layers:")
current = x
for i, node in enumerate(graph.node):
    if node.op_type == 'AveragePool':
        print(f"\nNode {i}: {node.op_type}")
        print(f"  Input shape: {current.shape}")
        
        # Run just this node
        # Create a sub-model with just this node for testing
        # Actually, let's just run the full model and check intermediate values
        
        # For now, let's check what the output shape should be
        # Input to AvgPool should be after ReLU and Pad
        # Let's trace: Conv -> ReLU -> Pad -> AvgPool
        
        # From earlier debug: ReLU output is 27x27x32 (in CORA format)
        # In ONNX format (BCSS): [1, 32, 27, 27]
        # After Pad (if any): still [1, 32, 27, 27] or [1, 32, 28, 28]?
        # After AvgPool: should be [1, 32, 6, 6]
        
        # If input is [1, 32, 27, 27] and output is [1, 32, 6, 6]
        # With kernel [4, 4] and stride [4, 4]: (27-4)/4 + 1 = 6.75 -> 6 (with floor)
        # But that requires padding or different kernel
        
        # Let's check: if input is [1, 32, 28, 28] (after padding)
        # With kernel [4, 4] and stride [4, 4]: (28-4)/4 + 1 = 7, not 6
        
        # If kernel [5, 5] and stride [4, 4]: (28-5)/4 + 1 = 6.75 -> 6
        # Or kernel [4, 4] and stride [5, 5]: (28-4)/5 + 1 = 5.8 -> 5, not 6
        
        # Let's try kernel [4, 4] with stride [4, 4] and padding [0, 0, 1, 1] (add 1 pixel)
        # Input becomes [1, 32, 28, 28], output: (28-4)/4 + 1 = 7, still not 6
        
        print(f"  Checking possible configurations...")

# Actually, let's just run ONNX Runtime on a simple input and see what happens
print("\n\nRunning ONNX Runtime on test input:")
x_test = np.ones((1, 1, 28, 28), dtype=np.float32)
outputs = session.run(None, {input_name: x_test})
print(f"Output shape: {outputs[0].shape}")
print(f"Output (first 10): {outputs[0].flatten()[:10]}")

