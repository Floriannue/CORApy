"""
Direct comparison of eAt values used in actual computation
"""
import numpy as np
from scipy.linalg import expm
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys.linearize import linearize
from cora_python.contDynamics.linearSys.oneStep import oneStep
from cora_python.contDynamics.contDynamics.linReach import linReach

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

print("=== eAt VALUE COMPARISON ===\n")

# Method 1: Direct expm
eAt_direct = expm(A * timeStep)
print("Method 1: Direct expm")
print(f"eAt[0, :]: {eAt_direct[0, :]}")
print(f"eAt diagonal: {np.diag(eAt_direct)}")

# Method 2: From taylorLinSys.getTaylor
if hasattr(linsys, 'taylor') and hasattr(linsys.taylor, 'getTaylor'):
    eAt_getTaylor = linsys.taylor.getTaylor('eAdt', timeStep=timeStep)
    print("\nMethod 2: taylorLinSys.getTaylor")
    print(f"eAt[0, :]: {eAt_getTaylor[0, :]}")
    print(f"eAt diagonal: {np.diag(eAt_getTaylor)}")
    
    diff = eAt_direct - eAt_getTaylor
    print(f"\nDifference (direct - getTaylor):")
    print(f"Max abs diff: {np.max(np.abs(diff))}")
    print(f"Are they identical? {np.allclose(eAt_direct, eAt_getTaylor, atol=1e-15)}")
    if not np.allclose(eAt_direct, eAt_getTaylor, atol=1e-15):
        print(f"Difference matrix:\n{diff}")

# Method 3: From homogeneousSolution (what's actually used)
print("\nMethod 3: From homogeneousSolution (actual usage)")
Rdelta = params['R0'] + (-sys.linError.p.x)
Htp, Hti, C_state = linsys.homogeneousSolution(Rdelta, timeStep, options['taylorTerms'])

# Extract eAt from the computation by checking what was used
if hasattr(linsys, 'taylor') and hasattr(linsys.taylor, 'eAt'):
    eAt_used = linsys.taylor.eAt
    print(f"eAt[0, :]: {eAt_used[0, :]}")
    print(f"eAt diagonal: {np.diag(eAt_used)}")
    
    diff_used = eAt_direct - eAt_used
    print(f"\nDifference (direct - used):")
    print(f"Max abs diff: {np.max(np.abs(diff_used))}")
    print(f"Are they identical? {np.allclose(eAt_direct, eAt_used, atol=1e-15)}")

# Method 4: Check what oneStep uses
print("\nMethod 4: From oneStep (full computation)")
Rtp, Rti, _, _, PU, Pu, _, C_input = oneStep(
    linsys, Rdelta, linParams['U'], linParams['uTrans'], 
    timeStep, options['taylorTerms']
)

# Check eAt after oneStep
if hasattr(linsys, 'taylor') and hasattr(linsys.taylor, 'eAt'):
    eAt_after_oneStep = linsys.taylor.eAt
    print(f"eAt[0, :]: {eAt_after_oneStep[0, :]}")
    print(f"eAt diagonal: {np.diag(eAt_after_oneStep)}")
    
    diff_after = eAt_direct - eAt_after_oneStep
    print(f"\nDifference (direct - after oneStep):")
    print(f"Max abs diff: {np.max(np.abs(diff_after))}")
    print(f"Are they identical? {np.allclose(eAt_direct, eAt_after_oneStep, atol=1e-15)}")

# Expected MATLAB values (from earlier debug output)
print("\n=== Expected MATLAB Values ===")
print("These should match Method 1 (direct expm) if implementations are identical")
print("If there are differences, they indicate:")
print("  1. Different expm implementations")
print("  2. Different floating-point precision")
print("  3. Different matrix multiplication order")

# Save values for MATLAB comparison
print("\n=== Values to compare with MATLAB ===")
print(f"A matrix:\n{A}")
print(f"\nA * timeStep:\n{A * timeStep}")
print(f"\neAt (full matrix):\n{eAt_direct}")
print(f"\neAt[0, :] (first row): {eAt_direct[0, :]}")
print(f"eAt diagonal: {np.diag(eAt_direct)}")
