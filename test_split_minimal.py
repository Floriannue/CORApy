"""
Minimal test to verify the splitting logic is correct
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from cora_python.nn.neuralNetwork.verify_helpers import _aux_split_with_dim

# Test case: split dimension 1 (0-indexed) with nSplits=2
xi = np.array([[0.64], [0.0], [0.0], [0.475], [-0.475]], dtype=np.float32)
ri = np.array([[0.04], [0.5], [0.5], [0.025], [0.025]], dtype=np.float32)

# Heuristic: dimension 1 (0-indexed, which is dimension 2 in 1-based) has highest value
his = np.array([[0.1], [1.0], [0.5], [0.05], [0.05]], dtype=np.float32)

print("Input:")
print(f"  xi = {xi.flatten()}")
print(f"  ri = {ri.flatten()}")
print(f"  his = {his.flatten()}")

# Debug: Check sorting
sortDims = np.argsort(np.abs(his), axis=0)[::-1]
print(f"\nDebug sorting:")
print(f"  np.argsort(his, axis=0) = {np.argsort(np.abs(his), axis=0).flatten()}")
print(f"  sortDims (descending) = {sortDims.flatten()}")
print(f"  sortDims[0, :] = {sortDims[0, :]}")
print(f"  Expected: sortDims[0, :] should be [1] (0-based index of max value 1.0)")
print(f"  As 1-based (dimId): should be [2]")

xis, ris, dimId = _aux_split_with_dim(xi, ri, his, nSplits=2)

print("\nOutput:")
print(f"  xis shape = {xis.shape}")
print(f"  xis[:,0] = {xis[:,0]}")  # First split
print(f"  xis[:,1] = {xis[:,1]}")  # Second split
print(f"  ris[:,0] = {ris[:,0]}")
print(f"  ris[:,1] = {ris[:,1]}")
print(f"  dimId = {dimId}")

# Expected for dimension 1 (0-indexed):
# Original: xi[1]=0, ri[1]=0.5
# After split:
#   Split 1: xi[1]=-0.25, ri[1]=0.25  (bounds: [-0.5, 0])
#   Split 2: xi[1]=0.25, ri[1]=0.25   (bounds: [0, 0.5])

print("\nExpected:")
print(f"  xis[:,0] should be [0.64, -0.25, 0.0, 0.475, -0.475]")
print(f"  xis[:,1] should be [0.64, 0.25, 0.0, 0.475, -0.475]")
print(f"  ris[:,0] should be [0.04, 0.25, 0.5, 0.025, 0.025]")
print(f"  ris[:,1] should be [0.04, 0.25, 0.5, 0.025, 0.025]")

# Verify
expected_xis_0 = np.array([0.64, -0.25, 0.0, 0.475, -0.475], dtype=np.float32)
expected_xis_1 = np.array([0.64, 0.25, 0.0, 0.475, -0.475], dtype=np.float32)
expected_ris = np.array([0.04, 0.25, 0.5, 0.025, 0.025], dtype=np.float32)

print("\nVerification:")
print(f"  xis[:,0] matches: {np.allclose(xis[:,0], expected_xis_0)}")
print(f"  xis[:,1] matches: {np.allclose(xis[:,1], expected_xis_1)}")
print(f"  ris[:,0] matches: {np.allclose(ris[:,0], expected_ris)}")
print(f"  ris[:,1] matches: {np.allclose(ris[:,1], expected_ris)}")

if not (np.allclose(xis[:,0], expected_xis_0) and np.allclose(xis[:,1], expected_xis_1)):
    print("\nERROR: Split results don't match expected values!")
    print("This indicates a bug in the splitting logic.")
else:
    print("\nSUCCESS: Split results match expected values!")

