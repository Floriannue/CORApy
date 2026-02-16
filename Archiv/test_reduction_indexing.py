"""test_reduction_indexing - Test the indexing logic in reduction"""

import numpy as np

print("=" * 80)
print("TESTING REDUCTION INDEXING")
print("=" * 80)

# Simulate the reduction logic
# Assume we have:
# - last0Idx = 0 (no zero generators)
# - nrG = 5 (5 generators total)
# - h = [0.1, 0.2, 0.3, 0.4, 0.5] (dH values for each generator)
# - dHmax = 0.25
# - idx = [2, 0, 4, 1, 3] (sorted indices of generators to reduce)

last0Idx = 0
nrG = 5
h = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
dHmax = 0.25
idx = np.array([2, 0, 4, 1, 3])

print(f"\nTest case:")
print(f"  last0Idx = {last0Idx}")
print(f"  nrG = {nrG}")
print(f"  h = {h}")
print(f"  dHmax = {dHmax}")
print(f"  idx = {idx}")

# MATLAB logic:
# redIdx = find(h <= dHmax, 1, 'last');
# This finds the last 1-based index where h <= dHmax
# h <= dHmax: [True, True, False, False, False]
# find(..., 1, 'last') returns 2 (1-based, meaning second element)

# Python equivalent:
redIdx_arr = np.where(h <= dHmax)[0]
print(f"\n  h <= dHmax: {h <= dHmax}")
print(f"  redIdx_arr (0-based): {redIdx_arr}")

if len(redIdx_arr) == 0:
    redIdx = 0
    print(f"  redIdx = 0 (no valid indices)")
else:
    redIdx_0based = redIdx_arr[-1]
    redIdx = redIdx_0based + 1  # Convert to 1-based
    print(f"  redIdx_0based (last 0-based): {redIdx_0based}")
    print(f"  redIdx (1-based, MATLAB style): {redIdx}")

# MATLAB: gredIdx = idx(1:length(hzeroIdx)+redIdx);
# This means: idx(1:0+2) = idx(1:2) = first 2 elements (1-based)
# In Python: idx[:0+2] = idx[:2] = first 2 elements (0-based)
gredIdx = idx[:last0Idx + redIdx]
print(f"\n  gredIdx = idx[:{last0Idx} + {redIdx}] = idx[:{last0Idx + redIdx}]")
print(f"  gredIdx = {gredIdx}")

# MATLAB: Gred = sum(gensred(:,1:redIdx),2);
# This means: sum columns 1 to redIdx (1-based, inclusive)
# So columns 0 to redIdx-1 in 0-based (redIdx columns total)
gensred = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])  # Example 2x5
print(f"\n  gensred shape: {gensred.shape}")
print(f"  gensred[:, :{redIdx}] (columns 0 to {redIdx-1}):")
print(f"  {gensred[:, :redIdx]}")
Gred = np.sum(gensred[:, :redIdx], axis=1, keepdims=True)
print(f"  Gred = sum(gensred[:, :{redIdx}], axis=1) = {Gred.flatten()}")

print("\n" + "=" * 80)
print("The key insight:")
print("  redIdx from MATLAB's find is 1-based index into h")
print("  It represents how many generators from gensred to reduce")
print("  So gensred(:,1:redIdx) means first redIdx columns")
print("=" * 80)
