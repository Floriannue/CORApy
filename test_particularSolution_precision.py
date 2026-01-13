"""
Test particularSolution_timeVarying precision improvements
Compare intermediate values with MATLAB
"""
import numpy as np
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys.linearize import linearize
from cora_python.contDynamics.linearSys.particularSolution_timeVarying import particularSolution_timeVarying

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

print("=" * 80)
print("TESTING particularSolution_timeVarying PRECISION")
print("=" * 80)

# Test particularSolution_timeVarying
U = linParams['U']
timeStep = options['timeStep']
truncationOrder = options['taylorTerms']

print(f"\nInput U:")
print(f"  U center: {U.c.flatten()}")
print(f"  U generators shape: {U.G.shape}")
print(f"  U first generator: {U.G[:, 0] if U.G.size > 0 else 'N/A'}")

# Compute particular solution
PU = particularSolution_timeVarying(linsys, U, timeStep, truncationOrder)

print(f"\nOutput PU:")
print(f"  PU center: {PU.c.flatten()}")
print(f"  PU generators shape: {PU.G.shape}")
print(f"  PU first generator: {PU.G[:, 0] if PU.G.size > 0 else 'N/A'}")
print(f"  PU generators (all):")
for i in range(min(5, PU.G.shape[1])):
    print(f"    Generator {i}: {PU.G[:, i]}")

# Check Taylor series computation
print(f"\nTaylor series computation check:")
print(f"  Using getTaylor: {hasattr(linsys, 'getTaylor')}")
if hasattr(linsys, 'taylor'):
    print(f"  taylor object exists: {linsys.taylor is not None}")
    if hasattr(linsys.taylor, 'dtoverfac'):
        print(f"  dtoverfac lists: {len(linsys.taylor.dtoverfac)}")
        if len(linsys.taylor.dtoverfac) > 0:
            idx = linsys.taylor.getIndexForTimeStep(timeStep)
            if idx != -1 and len(linsys.taylor.dtoverfac[idx]) > 0:
                print(f"  dtoverfac[1] (for timeStep={timeStep}): {linsys.taylor.dtoverfac[idx][0]}")
                if len(linsys.taylor.dtoverfac[idx]) > 1:
                    print(f"  dtoverfac[2]: {linsys.taylor.dtoverfac[idx][1]}")
                if len(linsys.taylor.dtoverfac[idx]) > 2:
                    print(f"  dtoverfac[3]: {linsys.taylor.dtoverfac[idx][2]}")

# Manual computation for first term
print(f"\nManual computation check:")
print(f"  First term: timeStep * U = {timeStep} * U")
PU_first = timeStep * U
print(f"  PU_first center: {PU_first.c.flatten()}")
print(f"  PU_first generators: {PU_first.G.shape}")

# Check if remainder term was added
print(f"\nRemainder term check:")
print(f"  truncationOrderInf: {np.isinf(truncationOrder)}")
if not np.isinf(truncationOrder):
    from cora_python.contDynamics.linearSys.private.priv_expmRemainder import priv_expmRemainder
    E = priv_expmRemainder(linsys, timeStep, truncationOrder)
    print(f"  E type: {type(E)}")
    print(f"  E is Interval: {hasattr(E, '__class__') and E.__class__.__name__ == 'Interval'}")

print("\n" + "=" * 80)
