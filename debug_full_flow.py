"""Debug the full flow of representsa_ for zonotope"""
import numpy as np
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope

# Test case from MATLAB
c = np.array([[2], [-1]])
G = np.array([[1], [-1]])
GI = np.array([[2, 0, 1], [-1, 1, 0]])
E = np.array([[3], [0]])

print("=== Test Case ===")
print(f"c: {c.flatten()}, dimension n = {c.shape[0]}")
print(f"G shape: {G.shape}")
print(f"GI shape: {GI.shape}")
print(f"E shape: {E.shape}")

pZ = PolyZonotope(c, G, GI, E)
print(f"\n=== After Construction ===")
print(f"pZ.E shape: {pZ.E.shape}, pZ.E:\n{pZ.E}")
print(f"pZ.G shape: {pZ.G.shape}, pZ.G:\n{pZ.G}")
print(f"pZ.GI shape: {pZ.GI.shape}")
print(f"pZ.G.size: {pZ.G.size}")

# Check compact_
print(f"\n=== After compact_ ===")
try:
    pZ_compact = pZ.compact_('all', np.finfo(float).eps)
    print(f"pZ_compact.E shape: {pZ_compact.E.shape}")
    print(f"pZ_compact.E:\n{pZ_compact.E}")
    print(f"pZ_compact.G shape: {pZ_compact.G.shape}")
    print(f"pZ_compact.G:\n{pZ_compact.G}")
    print(f"pZ_compact.G.size: {pZ_compact.G.size}")
    pZ = pZ_compact
except Exception as e:
    print(f"compact_ failed: {e}")

# Check representsa_ logic
print(f"\n=== representsa_('zonotope') Logic ===")
n = pZ.dim()
print(f"n = {n}")
print(f"res = n == 1 || aux_isZonotope(pZ,tol)")
print(f"res = {n} == 1 || aux_isZonotope(pZ,tol)")
print(f"res = False || aux_isZonotope(pZ,tol)")

# Check aux_isZonotope
print(f"\n=== aux_isZonotope Logic ===")
if pZ.G.size == 0:
    print("pZ.G is empty -> return True")
else:
    print("pZ.G is not empty -> continue")
    
    # removeRedundantExponents
    from cora_python.g.functions.helper.sets.contSet.polyZonotope.removeRedundantExponents import removeRedundantExponents
    E_new, G_new = removeRedundantExponents(pZ.E, pZ.G)
    print(f"\nAfter removeRedundantExponents:")
    print(f"E_new shape: {E_new.shape}, E_new:\n{E_new}")
    print(f"G_new shape: {G_new.shape}, G_new:\n{G_new}")
    
    # Dimension check
    print(f"\nDimension check:")
    print(f"MATLAB: if size(E,1) ~= size(G,2)")
    print(f"size(E_new,1) = {E_new.shape[0]}")
    print(f"size(G_new,2) = {G_new.shape[1]}")
    print(f"{E_new.shape[0]} ~= {G_new.shape[1]} -> {E_new.shape[0] != G_new.shape[1]}")
    if E_new.shape[0] != G_new.shape[1]:
        print("Check FAILS -> return False early")
    else:
        print("Check PASSES -> continue to identity check")
        
        # Sort and check identity
        if E_new.shape[1] == 1:
            E_sorted = E_new[np.argsort(-E_new.flatten())]
        else:
            sort_keys = [-E_new[:, i] for i in range(E_new.shape[1]-1, -1, -1)]
            E_sorted = E_new[np.lexsort(sort_keys)]
        print(f"\nE_sorted:\n{E_sorted}")
        
        diag_E = np.diag(np.diag(E_sorted))
        diff = np.abs(E_sorted - diag_E)
        sum_diff = np.sum(np.sum(diff))
        print(f"sum(sum(abs(E-diag_E))) = {sum_diff}")
        print(f"Is identity? {sum_diff == 0}")

# Check actual result
print(f"\n=== Actual Result ===")
result = pZ.representsa_('zonotope', tol=1e-10)
print(f"representsa_('zonotope') = {result}")

# Check what zonotope() returns
print(f"\n=== zonotope() Conversion ===")
try:
    Z = pZ.zonotope()
    print(f"Z.c:\n{Z.c}")
    print(f"Z.G:\n{Z.G}")
    print(f"Expected: zonotope(c,[G,GI])")
    from cora_python.contSet.zonotope.zonotope import Zonotope
    Z_expected = Zonotope(c, np.hstack([G, GI]))
    print(f"Expected Z.c:\n{Z_expected.c}")
    print(f"Expected Z.G:\n{Z_expected.G}")
    print(f"Match: {np.allclose(Z.c, Z_expected.c) and np.allclose(Z.G, Z_expected.G)}")
except Exception as e:
    print(f"zonotope() failed: {e}")

