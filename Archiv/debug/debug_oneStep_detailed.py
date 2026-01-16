"""
Detailed debugging of oneStep computation - the core of reachability
"""
import numpy as np
import math
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.center import center
from cora_python.contDynamics.nonlinearSys.linearize import linearize
from cora_python.contDynamics.linearSys.oneStep import oneStep

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

# Translate Rinit
Rdelta = params['R0'] + (-sys.linError.p.x)

print("=== DETAILED oneStep DEBUGGING ===\n")
print(f"Rdelta center: {Rdelta.c.flatten()}")
print(f"Rdelta generators shape: {Rdelta.G.shape}")
print(f"Rdelta generators sum(abs): {np.sum(np.abs(Rdelta.G), axis=1)}")

print(f"\nLinearized system A:\n{linsys.A}")
print(f"\nLinearized system B:\n{linsys.B}")

print(f"\nlinParams['U'] center: {center(linParams['U']).flatten()}")
print(f"linParams['U'] generators shape: {linParams['U'].G.shape}")

print(f"\nlinParams['uTrans'] type: {type(linParams['uTrans'])}")
if hasattr(linParams['uTrans'], 'c'):
    print(f"linParams['uTrans'] center: {linParams['uTrans'].c.flatten()}")
else:
    print(f"linParams['uTrans']: {linParams['uTrans']}")

# Get factors
options['factor'] = []
for i in range(1, options['taylorTerms'] + 2):
    options['factor'].append((options['timeStep'] ** i) / math.factorial(i))
print(f"\nFactors: {options['factor']}")

# Call oneStep
print("\n=== Calling oneStep ===")
Rtp, Rti, _, _, PU, Pu, _, C_input = oneStep(
    linsys, Rdelta, linParams['U'], linParams['uTrans'], 
    options['timeStep'], options['taylorTerms']
)

print(f"\n=== oneStep Results ===")
print(f"Rtp center: {Rtp.c.flatten()}")
print(f"Rtp generators shape: {Rtp.G.shape}")
print(f"Rtp generators sum(abs): {np.sum(np.abs(Rtp.G), axis=1)}")

print(f"\nRti center: {Rti.c.flatten()}")
print(f"Rti generators shape: {Rti.G.shape}")
print(f"Rti generators sum(abs): {np.sum(np.abs(Rti.G), axis=1)}")

# Check Taylor series computation
print(f"\n=== Taylor Series Check ===")
if hasattr(linsys, 'taylor') and hasattr(linsys.taylor, 'getTaylor'):
    eAt = linsys.taylor.getTaylor('eAdt', timeStep=options['timeStep'])
    print(f"eAt (exponential matrix) shape: {eAt.shape}")
    print(f"eAt (first row): {eAt[0, :]}")
    print(f"eAt (diagonal): {np.diag(eAt)}")
    
    # Check if eAt matches expected Taylor expansion
    A = linsys.A
    I = np.eye(A.shape[0])
    eAt_manual = I + A * options['timeStep']
    for i in range(1, options['taylorTerms']):
        eAt_manual += (A ** (i+1)) * (options['timeStep'] ** (i+1)) / math.factorial(i+1)
    print(f"\neAt manual (first row): {eAt_manual[0, :]}")
    print(f"eAt difference (first row): {(eAt[0, :] - eAt_manual[0, :])}")
