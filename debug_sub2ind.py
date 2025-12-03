"""
Debug sub2ind to understand the generator indexing issue
"""
import numpy as np

# Simulate the generator construction
Gxi_shape = (2, 2, 1)  # (n0, numInitGens, batch)
dimIdx = np.array([[1], [2]])  # 1-based: dimensions 1 and 2
numInitGens = 2
bSz = 1

print("=== sub2ind Debug ===\n")
print(f"Gxi shape: {Gxi_shape}")
print(f"dimIdx (1-based):\n{dimIdx}")
print(f"numInitGens: {numInitGens}, bSz: {bSz}\n")

# MATLAB: gIdx = sub2ind(size(Gxi),dimIdx, repmat((1:numInitGens)',1,bSz),repelem(1:bSz,numInitGens,1));
# This means:
# - dimIdx: which dimension (1-based)
# - repmat((1:numInitGens)',1,bSz): generator index (1-based)
# - repelem(1:bSz,numInitGens,1): batch index (1-based)

# For dimIdx = [[1], [2]]:
# Generator 0: (dim=1, gen=1, batch=1) -> should map to Gxi[0, 0, 0]
# Generator 1: (dim=2, gen=2, batch=1) -> should map to Gxi[1, 1, 0]

# Flatten column-major (MATLAB style)
dimIdx_flat = dimIdx.flatten('F')  # [1, 2]
genIdx_flat = np.tile(np.arange(1, numInitGens + 1), (1, bSz)).flatten('F')  # [1, 2]
batchIdx_flat = np.repeat(np.arange(1, bSz + 1), numInitGens)  # [1, 1]

print("Indices (1-based):")
print(f"  dimIdx_flat: {dimIdx_flat}")
print(f"  genIdx_flat: {genIdx_flat}")
print(f"  batchIdx_flat: {batchIdx_flat}")
print()

# Manual sub2ind calculation for 3D array
# sub2ind([n0, numGens, batch], dim, gen, batch)
# For shape (2, 2, 1):
# index = (dim-1) + (gen-1)*n0 + (batch-1)*n0*numGens
# index = (dim-1) + (gen-1)*2 + (batch-1)*2*2

gIdx_manual = []
for i in range(len(dimIdx_flat)):
    dim = dimIdx_flat[i] - 1  # Convert to 0-based
    gen = genIdx_flat[i] - 1
    batch = batchIdx_flat[i] - 1
    idx = dim + gen * Gxi_shape[0] + batch * Gxi_shape[0] * Gxi_shape[1]
    gIdx_manual.append(idx)
    print(f"  Generator {i}: (dim={dimIdx_flat[i]}, gen={genIdx_flat[i]}, batch={batchIdx_flat[i]}) -> linear idx {idx} -> Gxi[{dim}, {gen}, {batch}]")

print(f"\nLinear indices (0-based): {gIdx_manual}")
print()

# Now check what values should be set
ri = np.array([[1.0], [1.0]])
ri_dimIdx_flat = dimIdx.flatten('F')  # [1, 2]
ri_batchIdx_flat = np.repeat(np.arange(1, bSz + 1), numInitGens)  # [1, 1]

ri_gIdx_manual = []
for i in range(len(ri_dimIdx_flat)):
    dim = ri_dimIdx_flat[i] - 1
    batch = ri_batchIdx_flat[i] - 1
    idx = dim + batch * ri.shape[0]
    ri_gIdx_manual.append(idx)
    print(f"  ri index {i}: (dim={ri_dimIdx_flat[i]}, batch={ri_batchIdx_flat[i]}) -> linear idx {idx} -> ri[{dim}, {batch}] = {ri[dim, batch]}")

print(f"\nri linear indices (0-based): {ri_gIdx_manual}")
print(f"ri values: {[ri.flatten()[idx] for idx in ri_gIdx_manual]}")
print()

# Construct Gxi
Gxi = np.zeros(Gxi_shape)
Gxi_flat = Gxi.flatten()
for i, gidx in enumerate(gIdx_manual):
    ri_val = ri.flatten()[ri_gIdx_manual[i]]
    Gxi_flat[gidx] = ri_val
    print(f"  Setting Gxi_flat[{gidx}] = {ri_val}")

Gxi = Gxi_flat.reshape(Gxi_shape)
print(f"\nResulting Gxi[:,:numInitGens,:]:")
print(Gxi[:, :numInitGens, :])
print()
print("Expected:")
print("  Generator 0 should be in dimension 0: Gxi[0, 0, 0] = 1")
print("  Generator 1 should be in dimension 1: Gxi[1, 1, 0] = 1")
print(f"  Actual: Gxi[0, 0, 0] = {Gxi[0, 0, 0]}, Gxi[1, 1, 0] = {Gxi[1, 1, 0]}")

