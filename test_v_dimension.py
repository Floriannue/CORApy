import numpy as np
import sys
sys.path.insert(0, '.')

from cora_python.contDynamics.linearSys import LinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval

# Simulate the test case
N = 100
dim_x = 2 * N  # 200

# Create system like in test
A = np.block([[np.zeros((N, N)), np.eye(N)],
              [-np.eye(N), -np.eye(N)]])
C = np.zeros((1, 2 * N))
C[0, 0] = 1
B = np.ones((dim_x, 1))
sys = LinearSys('beam', A, B, None, C)

print(f"System outputs: {sys.nr_of_outputs}")
print(f"System noises: {sys.nr_of_noises}")
print(f"F shape: {sys.F.shape}")
print(f"F: {sys.F}")

# Check what _validateOptions sets for V
from cora_python.contDynamics.linearSys.reach import _validateOptions
params = {'tFinal': 0.01, 'R0': Zonotope(np.zeros((dim_x, 1)), np.zeros((dim_x, 0)))}
options = {}
params, options = _validateOptions(sys, params, options)

print(f"\nAfter _validateOptions:")
print(f"V type: {type(params['V'])}")
print(f"V dimension: {params['V'].dim()}")
print(f"V.c shape: {params['V'].c.shape}")

# Check if V is an Interval (which would trigger the conversion)
if isinstance(params['V'], Interval):
    print(f"\nV is an Interval, converting to zonotope...")
    v_zonotope = params['V'].zonotope()
    print(f"v_zonotope dimension: {v_zonotope.dim()}")
    print(f"v_zonotope.c shape: {v_zonotope.c.shape}")
    try:
        result = sys.F @ v_zonotope
        print(f"Success! Result dimension: {result.dim()}")
    except Exception as e:
        print(f"Error: {e}")
