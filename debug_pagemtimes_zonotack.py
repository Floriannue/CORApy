"""
Debug script to verify pagemtimes for zonotack attack computation
Compare Python einsum with MATLAB pagemtimes behavior
"""
import numpy as np

# Test case: zonotack attack computation
# Gxi_repeated: (n0, numInitGens, p*cbSz) = (2, 2, 1)
# beta: (numInitGens, 1, p*cbSz) = (2, 1, 1)

n0 = 2
numInitGens = 2
p = 1
cbSz = 1

# Create test arrays
Gxi_repeated = np.array([[[1.0], [0.5]], [[0.0], [1.0]]])  # (2, 2, 1)
beta = np.array([[[-1.0]], [[-1.0]]])  # (2, 1, 1)

print("=== Python pagemtimes for zonotack ===")
print(f"Gxi_repeated shape: {Gxi_repeated.shape}")
print(f"Gxi_repeated:\n{Gxi_repeated}")
print(f"\nbeta shape: {beta.shape}")
print(f"beta:\n{beta}")

# Current Python implementation
delta_python = np.einsum('ijk,jlk->ilk', Gxi_repeated, beta)
print(f"\nPython einsum('ijk,jlk->ilk') result shape: {delta_python.shape}")
print(f"delta_python:\n{delta_python}")

# What MATLAB pagemtimes should do:
# For each k: delta[:,:,k] = Gxi_repeated[:,:,k] @ beta[:,:,k]
# Gxi_repeated[:,:,0] is (2, 2), beta[:,:,0] is (2, 1)
# Result should be (2, 1)
delta_matlab_style = Gxi_repeated[:, :, 0] @ beta[:, :, 0]
print(f"\nMATLAB-style (manual): Gxi_repeated[:,:,0] @ beta[:,:,0]")
print(f"Result shape: {delta_matlab_style.shape}")
print(f"delta_matlab_style:\n{delta_matlab_style}")

# Check if they match
print(f"\nMatch: {np.allclose(delta_python.squeeze(), delta_matlab_style)}")

# Now test with actual values from the test case
print("\n=== With actual test values ===")
# From the test: xi=[0;0], ri=[1;1], so Gxi should have generators = ri
# If numInitGens=2, we have 2 generators, each set to ri for that dimension
Gxi_actual = np.array([[[1.0], [0.0]], [[0.0], [1.0]]])  # (2, 2, 1) - generators for dims 1 and 2
beta_actual = np.array([[[-1.0]], [[-1.0]]])  # (2, 1, 1)

print(f"Gxi_actual shape: {Gxi_actual.shape}")
print(f"Gxi_actual:\n{Gxi_actual}")
print(f"beta_actual shape: {beta_actual.shape}")
print(f"beta_actual:\n{beta_actual}")

delta_actual = np.einsum('ijk,jlk->ilk', Gxi_actual, beta_actual)
print(f"\ndelta_actual shape: {delta_actual.shape}")
print(f"delta_actual:\n{delta_actual}")

xi = np.array([[0.0], [0.0]])
zi_actual = xi + delta_actual.squeeze(1)
print(f"\nzi_actual = xi + delta_actual:")
print(f"xi: {xi.flatten()}")
print(f"delta_actual: {delta_actual.squeeze().flatten()}")
print(f"zi_actual: {zi_actual.flatten()}")

# Check bounds
ri = np.array([[1.0], [1.0]])
lower_bound = xi - ri
upper_bound = xi + ri
print(f"\nBounds: [{lower_bound[0,0]}, {upper_bound[0,0]}] x [{lower_bound[1,0]}, {upper_bound[1,0]}]")
print(f"zi_actual in bounds: {np.all(zi_actual >= lower_bound) and np.all(zi_actual <= upper_bound)}")

