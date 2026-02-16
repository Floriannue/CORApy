"""test_flatten_order - Test flattening order differences"""

import numpy as np

# Create test matrix
quadMat = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

print("Test matrix:")
print(quadMat)
print()

# MATLAB: quadMatoffdiag = quadMat + quadMat';
quadMatoffdiag = quadMat + quadMat.T
print("quadMat + quadMat.T:")
print(quadMatoffdiag)
print()

# MATLAB: quadMatoffdiag = quadMatoffdiag(:);  (column-major)
matlab_flatten = quadMatoffdiag.flatten(order='F')  # Column-major (Fortran order)
print("MATLAB flatten (column-major, order='F'):")
print(matlab_flatten)
print()

# Python default: row-major
python_flatten = quadMatoffdiag.flatten()  # Row-major (C order)
print("Python flatten (row-major, default):")
print(python_flatten)
print()

# Check if they're different
if not np.array_equal(matlab_flatten, python_flatten):
    print("DIFFERENT! This is the bug!")
    print(f"Difference: {matlab_flatten - python_flatten}")
else:
    print("Same")

# Test with lower triangular mask
gens = 2
kInd = np.tril(np.ones((gens+1, gens+1), dtype=bool), -1)
print(f"\nLower triangular mask (kInd):")
print(kInd)
print(f"kInd.flatten() (row-major): {kInd.flatten()}")
print(f"kInd.flatten(order='F') (column-major): {kInd.flatten(order='F')}")

# MATLAB: G(i, gens+1:end) = quadMatoffdiag(kInd(:));
# MATLAB uses column-major for both
matlab_result = matlab_flatten[kInd.flatten(order='F')]
print(f"\nMATLAB result (column-major for both): {matlab_result}")

# Python current (row-major for both)
python_result = python_flatten[kInd.flatten()]
print(f"Python result (row-major for both): {python_result}")

if not np.array_equal(matlab_result, python_result):
    print("DIFFERENT! This causes the bug!")
else:
    print("Same")
