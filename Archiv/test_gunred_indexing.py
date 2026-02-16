"""Test Gunred indexing logic"""
import numpy as np

print("=" * 80)
print("TESTING GUNRED INDEXING")
print("=" * 80)

# Test case: 5 generators, redIdx = 3
nrG = 5
last0Idx = 0
redIdx = 3  # 1-based: we reduce 3 generators

# Simulate idx (sorted indices of generators)
idx = np.array([0, 1, 2, 3, 4])  # 0-based indices

print(f"\nTest case:")
print(f"  nrG: {nrG}")
print(f"  last0Idx: {last0Idx}")
print(f"  redIdx (1-based): {redIdx}")
print(f"  idx: {idx}")

# MATLAB: Gunred = G(:,sort(idx(last0Idx+redIdx+1:end)));
# idx(last0Idx+redIdx+1:end) in 1-based means:
#   Start at position (last0Idx+redIdx+1) = (0+3+1) = 4 (1-based) = index 3 (0-based)
#   So: idx(4:end) in 1-based = idx[3:] in 0-based

matlab_start = last0Idx + redIdx + 1  # 1-based position
matlab_start_0based = matlab_start - 1  # Convert to 0-based
matlab_gunred_idx = idx[matlab_start_0based:]
print(f"\nMATLAB logic:")
print(f"  Start position (1-based): {matlab_start}")
print(f"  Start index (0-based): {matlab_start_0based}")
print(f"  gunred_idx: {matlab_gunred_idx}")

# Python: gunred_idx = idx[last0Idx + redIdx:]
# idx[last0Idx + redIdx:] in 0-based means:
#   Start at position (last0Idx + redIdx) = (0 + 3) = 3 (0-based)
#   So: idx[3:]

python_start = last0Idx + redIdx  # 0-based position
python_gunred_idx = idx[python_start:]
print(f"\nPython logic:")
print(f"  Start position (0-based): {python_start}")
print(f"  gunred_idx: {python_gunred_idx}")

print(f"\nMatch: {np.array_equal(matlab_gunred_idx, python_gunred_idx)}")

if not np.array_equal(matlab_gunred_idx, python_gunred_idx):
    print("\n[ERROR] Indexing mismatch!")
    print(f"  MATLAB: {matlab_gunred_idx}")
    print(f"  Python: {python_gunred_idx}")
else:
    print("\n[OK] Indexing matches")

# Test with different values
print("\n" + "=" * 80)
print("TESTING WITH DIFFERENT VALUES")
print("=" * 80)

test_cases = [
    (0, 1, [0, 1, 2, 3, 4]),  # last0Idx=0, redIdx=1
    (0, 2, [0, 1, 2, 3, 4]),  # last0Idx=0, redIdx=2
    (0, 3, [0, 1, 2, 3, 4]),  # last0Idx=0, redIdx=3
    (1, 2, [0, 1, 2, 3, 4]),  # last0Idx=1, redIdx=2
]

for last0Idx_test, redIdx_test, idx_test in test_cases:
    idx_arr = np.array(idx_test)
    matlab_start = last0Idx_test + redIdx_test + 1
    matlab_start_0based = matlab_start - 1
    matlab_result = idx_arr[matlab_start_0based:] if matlab_start_0based < len(idx_arr) else np.array([])
    
    python_start = last0Idx_test + redIdx_test
    python_result = idx_arr[python_start:] if python_start < len(idx_arr) else np.array([])
    
    match = np.array_equal(matlab_result, python_result)
    print(f"\nlast0Idx={last0Idx_test}, redIdx={redIdx_test}: Match={match}")
    if not match:
        print(f"  MATLAB: {matlab_result}")
        print(f"  Python: {python_result}")
