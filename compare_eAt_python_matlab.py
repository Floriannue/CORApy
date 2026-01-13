"""
Compare eAt values between Python and MATLAB
Uses the A matrix directly (no system setup required)
"""
import numpy as np
from scipy.linalg import expm
import math

# A matrix (from actual computation)
A = np.array([
    [-0.0234201,   0,          0,          0,          0,         -0.01],
    [0.0234201,  -0.01677445,  0,          0,          0,          0],
    [0,          0.01677445, -0.01661043,  0,          0,          0],
    [0,          0,          0.01661043, -0.02304648,  0,          0],
    [0,          0,          0,          0.02304648, -0.01062954,  0],
    [0,          0,          0,          0,          0.01062954, -0.01629874]
])

timeStep = 4

print("=== PYTHON eAt COMPUTATION ===\n")
print("A matrix:")
print(A)
print(f"\nTime step: {timeStep}\n")

# Method 1: Direct expm
print("=== Method 1: scipy.linalg.expm ===")
eAt_expm = expm(A * timeStep)
print("eAt from expm:")
print(eAt_expm)
print(f"eAt[0, :] (first row): {eAt_expm[0, :]}")
print(f"eAt diagonal: {np.diag(eAt_expm)}")

# Method 2: Manual Taylor series (4 terms)
print("\n=== Method 2: Manual Taylor Series (4 terms) ===")
I = np.eye(A.shape[0])
eAt_taylor = I.copy()
A_power = I.copy()
taylorTerms = 4

for i in range(1, taylorTerms + 1):
    A_power = A_power @ A
    term = A_power * (timeStep ** i) / math.factorial(i)
    eAt_taylor += term

print(f"eAt from Taylor (4 terms)[0, :]: {eAt_taylor[0, :]}")
print(f"eAt from Taylor (4 terms) diagonal: {np.diag(eAt_taylor)}")

diff_taylor = eAt_expm - eAt_taylor
print(f"Difference (expm - Taylor 4 terms) max abs: {np.max(np.abs(diff_taylor))}")

# Method 3: Extended Taylor series (20 terms)
print("\n=== Method 3: Extended Taylor Series (20 terms) ===")
eAt_taylor_ext = I.copy()
A_power_ext = I.copy()
for i in range(1, 21):
    A_power_ext = A_power_ext @ A
    term = A_power_ext * (timeStep ** i) / math.factorial(i)
    eAt_taylor_ext += term

print(f"eAt from Taylor (20 terms)[0, :]: {eAt_taylor_ext[0, :]}")
print(f"eAt from Taylor (20 terms) diagonal: {np.diag(eAt_taylor_ext)}")

diff_taylor_ext = eAt_expm - eAt_taylor_ext
print(f"Difference (expm - Taylor 20 terms) max abs: {np.max(np.abs(diff_taylor_ext))}")

# Matrix properties
print("\n=== Matrix Properties ===")
print(f"A norm (Frobenius): {np.linalg.norm(A, 'fro')}")
print(f"A norm (inf): {np.linalg.norm(A, np.inf)}")
print(f"A*timeStep norm (Frobenius): {np.linalg.norm(A * timeStep, 'fro')}")
print(f"A*timeStep norm (inf): {np.linalg.norm(A * timeStep, np.inf)}")

# Eigenvalues
eigenvals = np.linalg.eigvals(A)
print(f"\nA eigenvalues: {eigenvals}")
print(f"A*timeStep eigenvalues: {eigenvals * timeStep}")
eAt_eigenvals = np.linalg.eigvals(eAt_expm)
print(f"exp(A*timeStep) eigenvalues: {eAt_eigenvals}")
print(f"exp(A*timeStep eigenvalues) directly: {np.exp(eigenvals * timeStep)}")

# Output for MATLAB comparison
print("\n=== VALUES FOR MATLAB COMPARISON ===")
print("MATLAB should match these values:")
print(f"eAt[1, :] (MATLAB indexing, Python row 0): {eAt_expm[0, :]}")
print(f"eAt diagonal: {np.diag(eAt_expm)}")

# Save to file for easy comparison
print("\n=== SAVING TO FILE FOR COMPARISON ===")
with open('eAt_python_values.txt', 'w') as f:
    f.write("Python eAt values:\n")
    f.write(f"eAt[0, :]: {eAt_expm[0, :]}\n")
    f.write(f"eAt diagonal: {np.diag(eAt_expm)}\n")
    f.write("\nFull eAt matrix:\n")
    f.write(str(eAt_expm))
print("Saved to eAt_python_values.txt")
