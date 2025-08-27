import numpy as np

# Test the matrix dimensions that are causing the error
print("Testing matrix dimensions...")

# Simulate the weight matrix from the debug output
W = np.random.rand(50, 5)  # Shape (50, 5) as shown in debug
print(f"Weight matrix W shape: {W.shape}")

# Simulate the input data
input_data = np.random.rand(5, 1)  # Shape (5, 1) as shown in debug
print(f"Input data shape: {input_data.shape}")

# Test matrix multiplication
try:
    result = W @ input_data
    print(f"Matrix multiplication successful! Result shape: {result.shape}")
except Exception as e:
    print(f"Matrix multiplication failed: {e}")

# Test with different input shapes
print("\nTesting different input shapes:")
for shape in [(5, 1), (5,), (1, 5), (5, 2)]:
    test_input = np.random.rand(*shape)
    print(f"Input shape {shape}: ", end="")
    try:
        if len(shape) == 1:
            # Reshape to column vector
            test_input = test_input.reshape(-1, 1)
        result = W @ test_input
        print(f"✓ Success, result shape: {result.shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\nThe issue might be that the input data is not in the expected format.")
print("Let's check what the actual input data looks like in the example...")
