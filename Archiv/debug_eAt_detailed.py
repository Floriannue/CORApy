"""
Detailed investigation of eAt (exponential matrix) computation
"""
import numpy as np
import math
from scipy.linalg import expm
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.center import center
from cora_python.contDynamics.nonlinearSys.linearize import linearize

# Setup
dim_x = 6
params = {
    'R0': Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x)),
    'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1)),
    'tFinal': 4,
    'uTrans': np.zeros((1, 1))
}

options = {
    'timeStep': 4,
    'taylorTerms': 4,
    'zonotopeOrder': 50,
    'alg': 'lin',
    'tensorOrder': 2,
    'maxError': np.full((dim_x, 1), np.inf)
}

# System
tank = NonlinearSys(tank6Eq, states=6, inputs=1)

# Compute derivatives
from cora_python.contDynamics.contDynamics.derivatives import derivatives
derivatives(tank, options)

# Linearize
sys, linsys, linParams, linOptions = linearize(tank, params['R0'], params, options)

A = linsys.A
timeStep = options['timeStep']
taylorTerms = options['taylorTerms']

print("=== DETAILED eAt INVESTIGATION ===\n")
print(f"System matrix A:\n{A}")
print(f"\nTime step: {timeStep}")
print(f"Taylor terms: {taylorTerms}")

# Method 1: Direct expm computation (what Python uses)
print("\n=== Method 1: scipy.linalg.expm ===")
eAt_expm = expm(A * timeStep)
print(f"eAt from expm:\n{eAt_expm}")
print(f"\neAt[0, :]: {eAt_expm[0, :]}")
print(f"eAt diagonal: {np.diag(eAt_expm)}")

# Method 2: Manual Taylor series expansion
print("\n=== Method 2: Manual Taylor Series ===")
I = np.eye(A.shape[0])
eAt_taylor = I.copy()
A_power = I.copy()

print("Taylor series terms:")
for i in range(1, taylorTerms + 1):
    A_power = A_power @ A
    term = A_power * (timeStep ** i) / math.factorial(i)
    eAt_taylor += term
    print(f"  Term {i}: factor = {timeStep ** i / math.factorial(i):.10f}")
    print(f"    A^{i} * dt^{i}/{i}! (first row): {term[0, :]}")

print(f"\neAt from Taylor (first {taylorTerms} terms):\n{eAt_taylor}")
print(f"eAt_taylor[0, :]: {eAt_taylor[0, :]}")
print(f"eAt_taylor diagonal: {np.diag(eAt_taylor)}")

# Method 3: Extended Taylor series (more terms)
print("\n=== Method 3: Extended Taylor Series (20 terms) ===")
eAt_taylor_ext = I.copy()
A_power_ext = I.copy()
for i in range(1, 21):
    A_power_ext = A_power_ext @ A
    term = A_power_ext * (timeStep ** i) / math.factorial(i)
    eAt_taylor_ext += term
print(f"eAt_taylor_ext[0, :]: {eAt_taylor_ext[0, :]}")
print(f"eAt_taylor_ext diagonal: {np.diag(eAt_taylor_ext)}")

# Compare differences
print("\n=== Comparison ===")
diff_expm_taylor = eAt_expm - eAt_taylor
diff_expm_taylor_ext = eAt_expm - eAt_taylor_ext
print(f"Difference (expm - Taylor {taylorTerms} terms)[0, :]: {diff_expm_taylor[0, :]}")
print(f"Difference (expm - Taylor 20 terms)[0, :]: {diff_expm_taylor_ext[0, :]}")
print(f"Max abs diff (expm - Taylor {taylorTerms}): {np.max(np.abs(diff_expm_taylor))}")
print(f"Max abs diff (expm - Taylor 20): {np.max(np.abs(diff_expm_taylor_ext))}")

# Check what getTaylor returns
print("\n=== Method 4: getTaylor (from taylorLinSys) ===")
if hasattr(linsys, 'taylor') and hasattr(linsys.taylor, 'getTaylor'):
    eAt_getTaylor = linsys.taylor.getTaylor('eAdt', timeStep=timeStep)
    print(f"eAt from getTaylor:\n{eAt_getTaylor}")
    print(f"eAt_getTaylor[0, :]: {eAt_getTaylor[0, :]}")
    print(f"eAt_getTaylor diagonal: {np.diag(eAt_getTaylor)}")
    
    diff_expm_getTaylor = eAt_expm - eAt_getTaylor
    print(f"\nDifference (expm - getTaylor)[0, :]: {diff_expm_getTaylor[0, :]}")
    print(f"Max abs diff (expm - getTaylor): {np.max(np.abs(diff_expm_getTaylor))}")
    print(f"Are expm and getTaylor identical? {np.allclose(eAt_expm, eAt_getTaylor, atol=1e-15)}")

# Check matrix properties
print("\n=== Matrix Properties ===")
print(f"A norm (Frobenius): {np.linalg.norm(A, 'fro')}")
print(f"A norm (inf): {np.linalg.norm(A, np.inf)}")
print(f"A*timeStep norm (Frobenius): {np.linalg.norm(A * timeStep, 'fro')}")
print(f"A*timeStep norm (inf): {np.linalg.norm(A * timeStep, np.inf)}")

# Check eigenvalues
eigenvals = np.linalg.eigvals(A)
print(f"\nA eigenvalues: {eigenvals}")
print(f"A*timeStep eigenvalues: {eigenvals * timeStep}")
print(f"exp(A*timeStep) eigenvalues (should be exp of above): {np.linalg.eigvals(eAt_expm)}")
print(f"exp(A*timeStep eigenvalues) directly: {np.exp(eigenvals * timeStep)}")
