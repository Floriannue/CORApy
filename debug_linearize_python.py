"""
Debug script to trace linearize computation
"""
import numpy as np
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
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

# Get Rinit
Rinit = Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x))

print("=== Python linearize Debug ===")
print(f"Rinit center: {Rinit.c.flatten()}")
print(f"Rinit generators shape: {Rinit.G.shape}")

# Call linearize
sys, linsys, linParams, linOptions = linearize(tank, Rinit, params, options)

print(f"\nLinearization point p.x: {sys.linError.p.x.flatten()}")
print(f"Linearization point p.u: {sys.linError.p.u.flatten()}")
print(f"f0 (constant input): {sys.linError.f0.flatten()}")

print(f"\nLinearized system A shape: {linsys.A.shape}")
print(f"Linearized system A:\n{linsys.A}")
print(f"\nLinearized system B shape: {linsys.B.shape}")
print(f"Linearized system B:\n{linsys.B}")

# Test jacobian at linearization point
A_test, B_test = tank.jacobian(sys.linError.p.x, sys.linError.p.u)
print(f"\nJacobian A at p:\n{A_test}")
print(f"Jacobian B at p:\n{B_test}")
