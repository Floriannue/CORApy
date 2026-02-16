"""Test the reduction logic with concrete examples"""
import numpy as np

print("=" * 80)
print("TESTING REDUCTION LOGIC")
print("=" * 80)

# Simulate the reduction scenario from Step 2 Run 2
# Python: Rhom_tp has 5 generators → Rend_tp has 2 generators (reduced 3)
# MATLAB: Rhom_tp has 5 generators → Rend_tp has 4 generators (reduced 1)

# Test case: 5 generators, need to see which ones get reduced
nrG = 5
last0Idx = 0  # Assume no zero generators for simplicity
n = 2  # 2D zonotope

# Simulate gensred (5 generators, 2 dimensions)
np.random.seed(42)  # For reproducibility
gensred = np.random.rand(n, nrG) * 0.01  # Small values

print(f"\nTest case:")
print(f"  n (dimensions): {n}")
print(f"  nrG (generators): {nrG}")
print(f"  last0Idx: {last0Idx}")
print(f"  gensred shape: {gensred.shape}")

# Compute mugensred
maxidx = np.argmax(gensred, axis=0)
maxval = np.max(gensred, axis=0)
nrG_red = nrG - last0Idx
mugensred = np.zeros((n, nrG_red), dtype=gensred.dtype)
cols = n * np.arange(nrG_red)
mugensred.flat[cols + maxidx] = maxval

print(f"\n  maxidx: {maxidx}")
print(f"  maxval: {maxval}")

# Compute gensdiag and h
gensdiag = np.cumsum(gensred - mugensred, axis=1)
h = 2 * np.linalg.norm(gensdiag, axis=0, ord=2)

print(f"\n  gensdiag shape: {gensdiag.shape}")
print(f"  h: {h}")
print(f"  h values: {h}")

# Test with different dHmax values
dHmax_values = [0.001, 0.005, 0.01, 0.02, 0.05]

print("\n" + "-" * 80)
print("TESTING DIFFERENT dHmax VALUES")
print("-" * 80)

for dHmax in dHmax_values:
    redIdx_arr = np.where(h <= dHmax)[0]
    if len(redIdx_arr) == 0:
        redIdx = 0
        num_reduced = 0
    else:
        redIdx_0based = redIdx_arr[-1]
        redIdx = redIdx_0based + 1  # Convert to 1-based
        num_reduced = redIdx  # Number of generators reduced from gensred
    
    print(f"\ndHmax = {dHmax:.6f}:")
    print(f"  h <= dHmax: {h <= dHmax}")
    print(f"  redIdx_arr: {redIdx_arr}")
    print(f"  redIdx (1-based): {redIdx}")
    print(f"  Number of generators reduced: {num_reduced}")
    print(f"  Final generators: {nrG - num_reduced + n}")  # Unreduced + diagonal

# Test with actual values that might cause issues
print("\n" + "=" * 80)
print("TESTING FLOATING POINT PRECISION")
print("=" * 80)

# Simulate a case where h values are very close to dHmax
h_test = np.array([7.952e-06, 8.299e-06, 0.001, 0.002, 0.003])
dHmax_test = 0.006948944389885456  # From previous analysis

print(f"\nTest case with actual-like values:")
print(f"  h: {h_test}")
print(f"  dHmax: {dHmax_test}")
print(f"  h <= dHmax: {h_test <= dHmax_test}")

redIdx_arr = np.where(h_test <= dHmax_test)[0]
if len(redIdx_arr) == 0:
    redIdx = 0
else:
    redIdx_0based = redIdx_arr[-1]
    redIdx = redIdx_0based + 1

print(f"  redIdx_arr: {redIdx_arr}")
print(f"  redIdx (1-based): {redIdx}")
print(f"  All h values are <= dHmax: {np.all(h_test <= dHmax_test)}")

# Check if there's a difference with tolerance
tolerance = 1e-12
print(f"\nWith tolerance {tolerance}:")
print(f"  h <= dHmax + tolerance: {h_test <= dHmax_test + tolerance}")
