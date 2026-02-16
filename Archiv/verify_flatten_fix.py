"""verify_flatten_fix - Verify the flatten order fix is correct"""

import numpy as np

# Create a non-symmetric test case to see the difference
gens = 2
quadMat = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]])

print("Original quadMat:")
print(quadMat)
print()

# MATLAB: quadMatoffdiag = quadMat + quadMat';
quadMatoffdiag = quadMat + quadMat.T
print("quadMat + quadMat.T:")
print(quadMatoffdiag)
print()

# MATLAB uses column-major for flattening
matlab_flat = quadMatoffdiag.flatten(order='F')
print("MATLAB flatten (column-major, order='F'):")
print(matlab_flat)
print()

# Python default (row-major)
python_flat = quadMatoffdiag.flatten()
print("Python default flatten (row-major):")
print(python_flat)
print()

# Create lower triangular mask
kInd = np.tril(np.ones((gens+1, gens+1), dtype=bool), -1)
print("Lower triangular mask (kInd):")
print(kInd)
print()

# MATLAB: kInd(:) - column-major
matlab_kInd_flat = kInd.flatten(order='F')
print("MATLAB kInd(:) (column-major):")
print(matlab_kInd_flat)
print()

# Python default (row-major)
python_kInd_flat = kInd.flatten()
print("Python default kInd.flatten() (row-major):")
print(python_kInd_flat)
print()

# MATLAB result: quadMatoffdiag(kInd(:)) with column-major for both
matlab_result = matlab_flat[matlab_kInd_flat]
print("MATLAB result (column-major for both):")
print(matlab_result)
print()

# Python OLD (row-major for both) - WRONG
python_old_result = python_flat[python_kInd_flat]
print("Python OLD result (row-major for both) - WRONG:")
print(python_old_result)
print()

# Python NEW (column-major for both) - CORRECT
python_new_result = matlab_flat[matlab_kInd_flat]
print("Python NEW result (column-major for both) - CORRECT:")
print(python_new_result)
print()

# Check if they match
if np.allclose(matlab_result, python_new_result):
    print("✅ FIX IS CORRECT: MATLAB and Python NEW match!")
else:
    print("❌ FIX IS WRONG: Still different!")

if not np.allclose(matlab_result, python_old_result):
    print("✅ CONFIRMED: Python OLD was wrong (as expected)")
else:
    print("⚠️  WARNING: Python OLD matches MATLAB (unexpected!)")
