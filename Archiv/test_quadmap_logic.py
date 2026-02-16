"""test_quadmap_logic - Test quadMap logic with known values"""

import numpy as np
from cora_python.contSet.zonotope import Zonotope

print("=" * 80)
print("TESTING quadMap LOGIC")
print("=" * 80)

# Create test zonotope: center [1, 0], generators [[1, 0.5], [0, 0.3]]
Z = Zonotope(np.array([[1.0], [0.0]]), np.array([[1.0, 0.5], [0.0, 0.3]]))
print("\n1. Test Zonotope:")
print(f"   Center: {Z.c.flatten()}")
print(f"   Generators shape: {Z.G.shape}")
print(f"   Generators:\n{Z.G}")

# Create test Hessian matrix (sparse-like, but dense for testing)
# H[0] should be 2x2 to match zonotope dimension
# H[0] = [[-3, 0], [0, 0]]
H = [np.array([[-3.0, 0.0], [0.0, 0.0]])]

print("\n2. Test Hessian H[0]:")
print(H[0])

# Manual computation to verify logic
Zmat = np.hstack([Z.c, Z.G])
print(f"\n3. Zmat = [c, G]:")
print(f"   Shape: {Zmat.shape}")
print(f"   Zmat:\n{Zmat}")

# Compute quadMat = Zmat' * H[0] * Zmat
quadMat = Zmat.T @ H[0] @ Zmat
print(f"\n4. quadMat = Zmat' * H[0] * Zmat:")
print(f"   Shape: {quadMat.shape}")
print(f"   quadMat:\n{quadMat}")

# Extract diagonal elements
gens = Z.G.shape[1]
print(f"\n5. Diagonal extraction (gens={gens}):")
print(f"   MATLAB: diag(quadMat(2:gens+1,2:gens+1))")
print(f"   Python: diag(quadMat[1:gens+1, 1:gens+1])")
quadMat_sub = quadMat[1:gens+1, 1:gens+1]
diag_vals = np.diag(quadMat_sub)
print(f"   quadMat_sub:\n{quadMat_sub}")
print(f"   Diagonal values: {diag_vals}")
print(f"   G[0, :gens] = 0.5 * diag = {0.5 * diag_vals}")

# Center calculation
print(f"\n6. Center calculation:")
print(f"   MATLAB: c(1,1) = quadMat(1,1) + sum(G(1,1:gens))")
print(f"   Python: c[0, 0] = quadMat[0, 0] + sum(G[0, :gens])")
c_val = quadMat[0, 0] + np.sum(0.5 * diag_vals)
print(f"   quadMat[0, 0] = {quadMat[0, 0]}")
print(f"   sum(G[0, :gens]) = {np.sum(0.5 * diag_vals)}")
print(f"   c[0, 0] = {c_val}")

# Off-diagonal elements
print(f"\n7. Off-diagonal elements:")
quadMatoffdiag = quadMat + quadMat.T
print(f"   quadMat + quadMat.T:\n{quadMatoffdiag}")

# MATLAB: column-major flattening
matlab_flat = quadMatoffdiag.flatten(order='F')
print(f"\n   MATLAB flatten (column-major, order='F'):")
print(f"   {matlab_flat}")

# Python old: row-major flattening
python_old_flat = quadMatoffdiag.flatten()
print(f"\n   Python OLD flatten (row-major):")
print(f"   {python_old_flat}")

# Lower triangular mask
kInd = np.tril(np.ones((gens+1, gens+1), dtype=bool), -1)
print(f"\n   Lower triangular mask (kInd):")
print(kInd)

# MATLAB: column-major for mask too
matlab_kInd_flat = kInd.flatten(order='F')
print(f"\n   MATLAB kInd(:) (column-major):")
print(f"   {matlab_kInd_flat}")

# Python old: row-major for mask
python_old_kInd_flat = kInd.flatten()
print(f"\n   Python OLD kInd.flatten() (row-major):")
print(f"   {python_old_kInd_flat}")

# Extract off-diagonal elements
matlab_offdiag = matlab_flat[matlab_kInd_flat]
python_old_offdiag = python_old_flat[python_old_kInd_flat]

print(f"\n   MATLAB result (column-major for both):")
print(f"   {matlab_offdiag}")

print(f"\n   Python OLD result (row-major for both):")
print(f"   {python_old_offdiag}")

if np.allclose(matlab_offdiag, python_old_offdiag):
    print(f"\n   [NOTE] Results match for this symmetric case")
else:
    print(f"\n   [WARNING] Results differ!")

# Test actual quadMap call
print(f"\n8. Testing actual quadMap call:")
from cora_python.contSet.zonotope.quadMap import quadMap
Zquad = quadMap(Z, H)
print(f"   Result zonotope center: {Zquad.c.flatten()}")
print(f"   Result zonotope generators shape: {Zquad.G.shape}")
print(f"   Result zonotope generators:\n{Zquad.G}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
