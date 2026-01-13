"""
Compare particularSolution_timeVarying with MATLAB
Extract exact values from MATLAB to compare
"""
import numpy as np
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys.linearize import linearize
from cora_python.contDynamics.linearSys.particularSolution_timeVarying import particularSolution_timeVarying

# Setup (matching MATLAB exactly)
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

# Compute particular solution
U = linParams['U']
timeStep = options['timeStep']
truncationOrder = options['taylorTerms']

PU = particularSolution_timeVarying(linsys, U, timeStep, truncationOrder)

print("=" * 80)
print("PARTICULARSOLUTION_TIMEVARYING COMPARISON")
print("=" * 80)

print(f"\nPU generators shape: {PU.G.shape}")
print(f"\nPU generators (all columns):")
for i in range(PU.G.shape[1]):
    print(f"  Column {i}: {PU.G[:, i]}")

# Compute delta from PU generators
from cora_python.g.functions.helper.precision.kahan_sum import kahan_sum_abs
delta_PU = kahan_sum_abs(PU.G, axis=1)
print(f"\nDelta from PU generators (sum of abs):")
print(f"  {delta_PU}")

# Expected from MATLAB (from previous investigation)
# The difference in final delta is ~1.5e-3, which comes from PU generators
print(f"\nExpected delta contribution from PU (estimated):")
print(f"  Current delta: {delta_PU}")
print(f"  Difference suggests PU generators differ by ~1.5e-3 in sum")

print("\n" + "=" * 80)
print("To get exact MATLAB values, run:")
print("  matlab -batch \"run('debug_particularSolution_matlab.m')\"")
print("=" * 80)
