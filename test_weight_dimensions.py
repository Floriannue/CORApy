import numpy as np

# Test the exact dimensions from the debug output
print("Testing weight matrix dimensions from debug output...")

# From debug output: First MatMul layer has weight shape (5, 50)
W = np.random.rand(5, 50)  # Shape (5, 50) as shown in debug
print(f"Weight matrix W shape: {W.shape}")

# Input data has shape (5, 1) as shown in debug
x = np.random.rand(5, 1)  # Shape (5, 1)
print(f"Input data x shape: {x.shape}")

# Test matrix multiplication W @ x
print(f"\nTesting W @ x:")
print(f"W shape: {W.shape}")
print(f"x shape: {x.shape}")
print(f"Expected result shape: ({W.shape[0]}, {x.shape[1]}) = ({W.shape[0]}, 1)")

try:
    result = W @ x
    print(f"✓ Matrix multiplication successful! Result shape: {result.shape}")
    print(f"  This means: {W.shape[0]} output features, {x.shape[1]} batch size")
except Exception as e:
    print(f"✗ Matrix multiplication failed: {e}")

# Now test what the network actually expects
print(f"\nWhat the network expects:")
print(f"W shape: {W.shape} means {W.shape[0]} output features, {W.shape[1]} input features")
print(f"x shape: {x.shape} means {x.shape[0]} features, {x.shape[1]} batch size")

if W.shape[1] != x.shape[0]:
    print(f"❌ DIMENSION MISMATCH!")
    print(f"   Weight matrix expects {W.shape[1]} input features")
    print(f"   Input data provides {x.shape[0]} features")
    print(f"   Need to either:")
    print(f"     1. Transpose weights: W.T @ x")
    print(f"     2. Reshape input: x.T")
    print(f"     3. Fix weight loading")

# Test the correct orientation
print(f"\nTesting correct orientation:")
print(f"Option 1: W.T @ x (transpose weights)")
try:
    result = W.T @ x
    print(f"✓ W.T @ x successful! Result shape: {result.shape}")
except Exception as e:
    print(f"✗ W.T @ x failed: {e}")

print(f"Option 2: W @ x.T (transpose input)")
try:
    result = W @ x.T
    print(f"✓ W @ x.T successful! Result shape: {result.shape}")
except Exception as e:
    print(f"✗ W @ x.T failed: {e}")

print(f"\nConclusion:")
print(f"The issue is that the weight matrix has shape {W.shape} but expects input with {W.shape[1]} features.")
print(f"The input data has shape {x.shape} with {x.shape[0]} features.")
print(f"This suggests the weights are loaded in the wrong orientation or there's a missing preprocessing step.")
