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
print(f"System inputs: {sys.nr_of_inputs}")
print(f"System states: {sys.nr_of_states}")
print(f"D shape: {sys.D.shape}")
print(f"F shape: {sys.F.shape}")

# Simulate what happens in aux_canonicalForm
from cora_python.contDynamics.linearSys.reach import _validateOptions
params = {
    'tFinal': 0.01, 
    'R0': Zonotope(np.zeros((dim_x, 1)), np.zeros((dim_x, 0))),
    'U': Zonotope(np.zeros((1, 1)), np.zeros((1, 0)))  # Input dimension
}
options = {}
params, options = _validateOptions(sys, params, options)

print(f"\nAfter _validateOptions:")
print(f"U dimension: {params['U'].dim()}")
print(f"V dimension: {params['V'].dim()}")

# Simulate line 189: params['U'] = linsys.B @ params['U'] + W
W = Zonotope(np.zeros((dim_x, 1)), np.zeros((dim_x, 0)))
params['U'] = sys.B @ params['U'] + W
print(f"\nAfter B @ U + W:")
print(f"U dimension: {params['U'].dim()}")

# Now check line 156: params['V'] = linsys.D @ params['U'] + params['V']
print(f"\nTrying D @ U + V:")
print(f"D shape: {sys.D.shape}")
print(f"U dimension: {params['U'].dim()}")
print(f"V dimension: {params['V'].dim()}")

try:
    # This should work: (1, 1) @ (200, ?) zonotope
    result = sys.D @ params['U']
    print(f"D @ U dimension: {result.dim()}")
    result2 = result + params['V']
    print(f"D @ U + V dimension: {result2.dim()}")
except Exception as e:
    print(f"Error in D @ U: {e}")

# Now check the F @ V issue
# If V becomes an Interval somehow with wrong dimension
print(f"\nChecking F @ V issue:")
print(f"F shape: {sys.F.shape}")
# Simulate if V was an Interval with dimension 200 (wrong!)
v_wrong = Interval(np.zeros(200), np.zeros(200))
v_zonotope = v_wrong.zonotope()
print(f"v_wrong dimension: {v_wrong.dim()}")
print(f"v_zonotope dimension: {v_zonotope.dim()}")
print(f"v_zonotope.c shape: {v_zonotope.c.shape}")
try:
    result = sys.F @ v_zonotope
    print(f"Success!")
except Exception as e:
    print(f"Error (expected): {e}")
