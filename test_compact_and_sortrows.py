"""Test compact_ and sortrows behavior"""
import numpy as np
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope

# Test case from the failing test
c = np.array([[2], [-1]])
G = np.array([[1], [-1]])
GI = np.array([[2, 0, 1], [-1, 1, 0]])
E = np.array([[3], [0]])

print("=== Original ===")
print(f"E:\n{E}")
print(f"E shape: {E.shape}")

pZ = PolyZonotope(c, G, GI, E)
print(f"\npZ.E:\n{pZ.E}")
print(f"pZ.E shape: {pZ.E.shape}")

# Test compact_
print("\n=== After compact_('all') ===")
pZ_compact = pZ.compact_('all', np.finfo(float).eps)
print(f"pZ_compact.E:\n{pZ_compact.E}")
print(f"pZ_compact.E shape: {pZ_compact.E.shape}")
print(f"pZ_compact.G:\n{pZ_compact.G}")
print(f"pZ_compact.G shape: {pZ_compact.G.shape}")

# Test sortrows
print("\n=== Testing sortrows ===")
E_test = np.array([[3], [0]])
print(f"E_test:\n{E_test}")

# MATLAB: sortrows(E,'descend') for a column vector
# This should sort rows in descending order
# For [3; 0], descending order is [3; 0] (already sorted)

# Test with multiple columns
E_test2 = np.array([[3, 1], [0, 2], [3, 0]])
print(f"\nE_test2:\n{E_test2}")
print("MATLAB sortrows(E_test2,'descend') would sort by:")
print("  First column descending: [3,1], [3,0], [0,2]")
print("  Then second column descending: [3,1], [3,0], [0,2]")

# Python implementation
if E_test2.shape[1] == 1:
    E_sorted = E_test2[np.argsort(-E_test2.flatten())]
else:
    sort_keys = [-E_test2[:, i] for i in range(E_test2.shape[1]-1, -1, -1)]
    E_sorted = E_test2[np.lexsort(sort_keys)]

print(f"\nPython sorted:\n{E_sorted}")

# Test identity check
print("\n=== Testing identity check ===")
E_identity = np.eye(2)
print(f"E_identity:\n{E_identity}")
diag_E = np.diag(np.diag(E_identity))
print(f"diag_E:\n{diag_E}")
diff = np.abs(E_identity - diag_E)
print(f"abs(E - diag_E):\n{diff}")
sum_diff = np.sum(np.sum(diff))
print(f"sum(sum(abs(E - diag_E))): {sum_diff}")
print(f"Is identity? {sum_diff == 0}")

# Test with compacted E
print("\n=== Testing with compacted E ===")
E_compacted = np.array([[3]])  # After compact_ removes zero row
print(f"E_compacted:\n{E_compacted}")
print(f"E_compacted shape: {E_compacted.shape}")
print(f"G shape: {pZ_compact.G.shape}")
print(f"size(E,1) = {E_compacted.shape[0]}")
print(f"size(G,2) = {pZ_compact.G.shape[1]}")
print(f"Check: {E_compacted.shape[0]} == {pZ_compact.G.shape[1]} -> {E_compacted.shape[0] == pZ_compact.G.shape[1]}")

# Sort
E_sorted_compacted = E_compacted[np.argsort(-E_compacted.flatten())]
print(f"\nE_sorted_compacted:\n{E_sorted_compacted}")

# Identity check
diag_E_compacted = np.diag(np.diag(E_sorted_compacted))
print(f"diag_E_compacted:\n{diag_E_compacted}")
diff_compacted = np.abs(E_sorted_compacted - diag_E_compacted)
print(f"abs(E - diag_E):\n{diff_compacted}")
sum_diff_compacted = np.sum(np.sum(diff_compacted))
print(f"sum(sum(abs(E - diag_E))): {sum_diff_compacted}")
print(f"Is identity? {sum_diff_compacted == 0}")

