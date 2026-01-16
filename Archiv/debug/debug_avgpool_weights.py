"""Debug script to check AvgPool weight construction"""
import numpy as np

# Simulate the weight construction (NEW FIXED VERSION)
p_h, p_w = 4, 4
in_c = 32

# New construction: Initialize with zeros and set diagonal blocks
W = np.zeros((p_h, p_w, in_c, in_c), dtype=np.float64)
pool_value = 1.0 / (p_h * p_w)
for i in range(in_c):
    W[:, :, i, i] = pool_value

print(f"W shape: {W.shape}")
print(f"W (first 2x2x2x2):\n{W[:2, :2, :2, :2]}")
print(f"\nExpected: all values should be 0.0625 (1/16)")
print(f"Actual min: {W.min()}, max: {W.max()}, mean: {W.mean()}")

# Check if each channel block is correct
print(f"\nChecking channel 0 block (should be all 0.0625):")
print(f"Channel 0 block:\n{W[:, :, 0, 0]}")
print(f"All values 0.0625? {np.allclose(W[:, :, 0, 0], 0.0625)}")

# Check channel 1 block
print(f"\nChecking channel 1 block (should be all 0.0625):")
print(f"Channel 1 block:\n{W[:, :, 1, 1]}")
print(f"All values 0.0625? {np.allclose(W[:, :, 1, 1], 0.0625)}")

# Check cross-channel (should be zeros)
print(f"\nChecking cross-channel [0, 1] (should be all zeros):")
print(f"Cross-channel block:\n{W[:, :, 0, 1]}")
print(f"All values zero? {np.allclose(W[:, :, 0, 1], 0)}")

