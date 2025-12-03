"""
Debug script to understand why zonotack attack produces out-of-bounds points
Focus on generator construction and how summing generators can exceed bounds
"""
import numpy as np

# Test case: xi=[0;0], ri=[1;1], numInitGens=2
xi = np.array([[0.0], [0.0]])
ri = np.array([[1.0], [1.0]])
numInitGens = 2
n0 = 2

print("=== Generator Construction Analysis ===\n")
print(f"Input: xi = {xi.flatten()}, ri = {ri.flatten()}")
print(f"numInitGens = {numInitGens}")
print(f"Bounds: [{xi[0,0]-ri[0,0]}, {xi[0,0]+ri[0,0]}] x [{xi[1,0]-ri[1,0]}, {xi[1,0]+ri[1,0]}]")
print()

# Simulate how Gxi is constructed (from _aux_constructInputZonotope)
# For numInitGens=2, we typically use the first numInitGens dimensions
# Each generator corresponds to one dimension: Gxi[i, i, :] = ri[i]
Gxi = np.zeros((n0, numInitGens, 1))
for i in range(numInitGens):
    Gxi[i, i, 0] = ri[i, 0]

print("Gxi (generators):")
print(f"Shape: {Gxi.shape}")
print(f"Gxi[:,:,0]:\n{Gxi[:,:,0]}")
print()

# Now simulate zonotack attack
# beta = -sign(ld_Gyi) where ld_Gyi has shape (p, numInitGens, cbSz)
# For simplicity, assume both generators have negative sign
p = 1
cbSz = 1

# beta_ = -sign(ld_Gyi) -> (p, numInitGens, cbSz)
# After permute and reshape: beta = (numInitGens, 1, p*cbSz)
# If both generators contribute negatively:
beta = np.array([[[-1.0]], [[-1.0]]])  # (2, 1, 1)
print("beta (attack coefficients):")
print(f"Shape: {beta.shape}")
print(f"beta:\n{beta}")
print()

# Gxi_repeated = repelem(Gxi(:,1:numInitGens,:),1,1,p)
Gxi_subset = Gxi[:, :numInitGens, :]  # (n0, numInitGens, cbSz)
Gxi_repeated = np.repeat(Gxi_subset, p, axis=2)  # (n0, numInitGens, p*cbSz)
print("Gxi_repeated:")
print(f"Shape: {Gxi_repeated.shape}")
print(f"Gxi_repeated[:,:,0]:\n{Gxi_repeated[:,:,0]}")
print()

# delta = pagemtimes(Gxi_repeated, beta)
# einsum('ijk,jlk->ilk', Gxi_repeated, beta)
delta = np.einsum('ijk,jlk->ilk', Gxi_repeated, beta)  # (n0, 1, p*cbSz)
delta = delta.squeeze(1)  # (n0, p*cbSz)
print("delta (attack vector):")
print(f"Shape: {delta.shape}")
print(f"delta:\n{delta}")
print(f"|delta|: {np.abs(delta).flatten()}")
print(f"ri: {ri.flatten()}")
print(f"|delta| > ri: {np.abs(delta).flatten() > ri.flatten()}")
print()

# zi = xi_repeated + delta
xi_repeated = np.repeat(xi, p, axis=1)  # (n0, p*cbSz)
zi = xi_repeated + delta
print("zi (attack point):")
print(f"zi: {zi.flatten()}")
print()

# Check bounds
lower_bound = xi - ri
upper_bound = xi + ri
in_bounds = np.all(zi >= lower_bound) and np.all(zi <= upper_bound)
print("Bounds check:")
print(f"Lower: {lower_bound.flatten()}")
print(f"Upper: {upper_bound.flatten()}")
print(f"zi in bounds: {in_bounds}")
if not in_bounds:
    print(f"  zi[0] = {zi[0,0]}, bounds: [{lower_bound[0,0]}, {upper_bound[0,0]}]")
    print(f"  zi[1] = {zi[1,0]}, bounds: [{lower_bound[1,0]}, {upper_bound[1,0]}]")
print()

print("=== Analysis ===")
print("The problem: When numInitGens > 1, we sum multiple generators.")
print("Each generator can contribute up to ri[i] in magnitude.")
print("If all generators have the same sign, the sum can exceed ri[i].")
print()
print("For example:")
print("  Generator 1: Gxi[0,0] = 1.0, beta[0] = -1 -> contributes -1.0")
print("  Generator 2: Gxi[0,1] = 0.0, beta[1] = -1 -> contributes 0.0")
print("  But if Gxi[0,1] != 0, then delta[0] could exceed ri[0]")
print()
print("The issue: Gxi generators might not be orthogonal/independent.")
print("If Gxi[0,1] != 0, then generator 2 also affects dimension 0.")
print("This can cause delta[0] to exceed ri[0] when both generators are active.")

