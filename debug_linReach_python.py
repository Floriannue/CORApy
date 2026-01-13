"""
Debug script to trace intermediate values in linReach computation
"""
import numpy as np
import math
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contDynamics.contDynamics.linReach import linReach

# Setup (same as test)
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

# Get factors
options['factor'] = []
for i in range(1, options['taylorTerms'] + 2):
    options['factor'].append((options['timeStep'] ** i) / math.factorial(i))

print("=== Python linReach Debug ===")
print(f"R0 center: {params['R0'].c.flatten()}")
print(f"R0 generators shape: {params['R0'].G.shape}")
print(f"factors: {options['factor']}")

# Call linReach
Rinit = {'set': params['R0'], 'error': np.zeros((dim_x, 1))}
Rti, Rtp, dimForSplit, opts = linReach(tank, Rinit, params, options)

print(f"\nRti type: {type(Rti)}")
if hasattr(Rti, 'c'):
    print(f"Rti center: {Rti.c.flatten()}")
    print(f"Rti generators shape: {Rti.G.shape}")

print(f"\nRtp type: {type(Rtp)}")
if isinstance(Rtp, dict):
    if 'set' in Rtp:
        print(f"Rtp['set'] type: {type(Rtp['set'])}")
        if hasattr(Rtp['set'], 'c'):
            print(f"Rtp['set'] center: {Rtp['set'].c.flatten()}")
            print(f"Rtp['set'] generators shape: {Rtp['set'].G.shape}")
    if 'error' in Rtp:
        print(f"Rtp['error']: {Rtp['error'].flatten()}")

# Get interval hulls
IH_ti = Interval(Rti)
IH_tp = Interval(Rtp['set'] if isinstance(Rtp, dict) else Rtp)

print(f"\nIH_ti inf: {IH_ti.inf.flatten()}")
print(f"IH_ti sup: {IH_ti.sup.flatten()}")
print(f"\nIH_tp inf: {IH_tp.inf.flatten()}")
print(f"IH_tp sup: {IH_tp.sup.flatten()}")
