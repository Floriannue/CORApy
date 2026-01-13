"""
Debug dimension mismatch in wrapping-free reachability
"""
import numpy as np
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.zonotope import Zonotope

# Create a simple test case
A = np.array([[-1, 0], [0, -2]])
B = np.array([[1], [0]])
sys = LinearSys('test', A, B)

# Parameters
params = {
    'R0': Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2)),
    'U': Zonotope(np.array([[1]]), 0.5 * np.array([[0.2]])),
    'tFinal': 0.04,
    'tStart': 0.0,
    'uTrans': np.array([[1]])
}

options = {
    'timeStep': 0.04,
    'taylorTerms': 4,
    'linAlg': 'wrapping-free'
}

# Test oneStep to see what dimensions we get
from cora_python.contDynamics.linearSys.oneStep import oneStep
Rtp, Rti, Htp, Hti, PU, Pu, _, C_input = oneStep(
    sys, params['R0'], params['U'], params['uTrans'], 
    options['timeStep'], options['taylorTerms'])

print("Dimensions after oneStep:")
print(f"  Hti: {Hti.dim()}")
print(f"  PU: {PU.dim()}, type: {type(PU)}")
print(f"  C_input: {C_input.dim()}, type: {type(C_input)}")

# Convert PU to Interval
from cora_python.contSet.interval import Interval
PU_interval = Interval(PU)
print(f"  PU_interval: {PU_interval.dim()}, type: {type(PU_interval)}")
print(f"  PU_interval.inf.shape: {PU_interval.inf.shape}")
print(f"  PU_interval.sup.shape: {PU_interval.sup.shape}")

# Try adding Hti + PU_interval
try:
    result1 = Hti + PU_interval
    print(f"  Hti + PU_interval: {result1.dim()}, type: {type(result1)}")
except Exception as e:
    print(f"  Error in Hti + PU_interval: {e}")

# Try adding result + C_input
try:
    result1 = Hti + PU_interval
    result2 = result1 + C_input
    print(f"  (Hti + PU_interval) + C_input: {result2.dim()}, type: {type(result2)}")
except Exception as e:
    print(f"  Error in (Hti + PU_interval) + C_input: {e}")
