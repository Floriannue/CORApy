"""
Debug script to check what Python's affineSolution returns for Pu
"""
import numpy as np
import sys
sys.path.insert(0, 'cora_python')

from cora_python.contDynamics.linearSys import LinearSys, affineSolution
from cora_python.contSet.zonotope import Zonotope
import scipy.linalg

# Same inputs as MATLAB test
A = np.array([[-1, -4], [4, -1]], dtype=float)
sys = LinearSys(A)

center = np.array([[40], [20]], dtype=float)
generators = np.array([[1, 4, 2], [-1, 3, 5]], dtype=float)
X = Zonotope(center, generators)

u = np.array([[2], [-1]], dtype=float)
timeStep = 0.05
truncationOrder = 6

# Compute reachable sets
Htp, Pu, Hti, C_state, C_input = affineSolution(sys, X, u, timeStep, truncationOrder)

print("Python affineSolution Output")
print("=" * 50)
print(f"Pu type: {type(Pu)}")
print(f"Pu = \n{Pu}")
if hasattr(Pu, 'center'):
    print(f"Pu.center() = \n{Pu.center()}")
if isinstance(Pu, list):
    print(f"Pu is a list with {len(Pu)} elements")
    for i, p in enumerate(Pu):
        print(f"Pu[{i}] type: {type(p)}")
        print(f"Pu[{i}] = \n{p}")

# Compare to analytical solution
Pu_true = np.linalg.inv(A) @ (scipy.linalg.expm(A * timeStep) - np.eye(2)) @ u
print(f"\nPu_true = \n{Pu_true}")

# Extract Pu_center for comparison
if isinstance(Pu, list):
    Pu_center = Pu[0].center() if hasattr(Pu[0], 'center') else Pu[0]
else:
    Pu_center = Pu.center() if hasattr(Pu, 'center') else Pu

print(f"\nPu_center (for comparison) = \n{Pu_center}")
print(f"Pu_center type: {type(Pu_center)}")

# Check difference
if isinstance(Pu_center, np.ndarray) and isinstance(Pu_true, np.ndarray):
    diff = np.abs(Pu_center - Pu_true)
    max_diff = np.max(diff)
    print(f"\nMax difference = {max_diff}")
    print(f"All close (1e-14): {np.allclose(Pu_center, Pu_true, atol=1e-14, rtol=0)}")
