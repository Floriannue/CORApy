"""
Step-by-step debug to compare with MATLAB
"""
import numpy as np
import sys
sys.path.insert(0, 'cora_python')

from cora_python.contDynamics.linearSys import LinearSys
from cora_python.contSet.zonotope import Zonotope
import scipy.linalg

# Same inputs as MATLAB test
A = np.array([[-1, -4], [4, -1]], dtype=float)
sys = LinearSys(A)

# Initialize taylor
if not hasattr(sys, 'taylor') or sys.taylor is None:
    from cora_python.g.classes.taylorLinSys import TaylorLinSys
    sys.taylor = TaylorLinSys(sys.A)

u = np.array([[2], [-1]], dtype=float)
timeStep = 0.05
truncationOrder = 6

# Convert u to zonotope
from cora_python.contSet.zonotope import Zonotope
U = Zonotope(u)
blocks = np.array([[1, 2]])

# Decompose
U_decomp = U.decompose(blocks)

print("=== Step-by-step computation ===\n")
print(f"U_decomp type: {type(U_decomp)}")
if hasattr(U_decomp, 'c'):
    print(f"U_decomp.c shape: {U_decomp.c.shape}")
    print(f"U_decomp.G shape: {U_decomp.G.shape}")

# Compute Asum
Asum = timeStep * np.eye(2)
print(f"\nAsum (initial) = \n{Asum}")

# Compute terms
for eta in range(1, truncationOrder + 1):
    Apower_mm = sys.taylor._computeApower(eta)
    import math
    dtoverfac = timeStep**(eta + 1) / math.factorial(eta + 1)
    addTerm = Apower_mm * dtoverfac
    Asum = Asum + addTerm
    if eta <= 3:
        print(f"\neta={eta}:")
        print(f"  Apower_mm = \n{Apower_mm}")
        print(f"  dtoverfac = {dtoverfac}")
        print(f"  addTerm = \n{addTerm}")
        print(f"  Asum = \n{Asum}")

print(f"\nFinal Asum = \n{Asum}")

# Compute remainder term
from cora_python.contDynamics.linearSys.private.priv_expmRemainder import priv_expmRemainder
E = priv_expmRemainder(sys, timeStep, truncationOrder)
print(f"\nE type: {type(E)}")
if hasattr(E, 'inf'):
    print(f"E.inf = \n{E.inf}")
    print(f"E.sup = \n{E.sup}")
    from cora_python.contSet.interval.center import center
    from cora_python.contSet.interval.rad import rad
    E_center = center(E)
    E_rad = rad(E)
    print(f"E.center() = \n{E_center}")
    print(f"E.rad() = \n{E_rad}")

# Compute E * timeStep
E_times_dt = E * timeStep
print(f"\nE * timeStep type: {type(E_times_dt)}")
if hasattr(E_times_dt, 'inf'):
    print(f"E * timeStep .inf = \n{E_times_dt.inf}")
    print(f"E * timeStep .sup = \n{E_times_dt.sup}")

# Compute block_mtimes(Asum, U_decomp)
from cora_python.g.functions.helper.sets.contSet.contSet.block_mtimes import block_mtimes
Ptp1 = block_mtimes(Asum, U_decomp)
print(f"\nPtp1 (Asum * U_decomp) type: {type(Ptp1)}")
if hasattr(Ptp1, 'c'):
    print(f"Ptp1.c = \n{Ptp1.c}")
    print(f"Ptp1.G shape: {Ptp1.G.shape}")

# Compute block_mtimes(E * timeStep, U_decomp)
Ptp2 = block_mtimes(E_times_dt, U_decomp)
print(f"\nPtp2 (E*timeStep * U_decomp) type: {type(Ptp2)}")
if hasattr(Ptp2, 'c'):
    print(f"Ptp2.c = \n{Ptp2.c}")
    print(f"Ptp2.G shape: {Ptp2.G.shape}")

# Sum
from cora_python.g.functions.helper.sets.contSet.contSet.block_operation import block_operation
Ptp = block_operation(lambda a, b: a + b, Ptp1, Ptp2)
print(f"\nPtp (sum) type: {type(Ptp)}")
if hasattr(Ptp, 'c'):
    print(f"Ptp.c = \n{Ptp.c}")

# Extract center
from cora_python.contDynamics.linearSys.particularSolution_constant import _center
Pu = block_operation(_center, Ptp)
print(f"\nPu (after center) = \n{Pu}")
print(f"Pu type: {type(Pu)}")

# Compare to analytical
Pu_true = np.linalg.inv(A) @ (scipy.linalg.expm(A * timeStep) - np.eye(2)) @ u
print(f"\nPu_true = \n{Pu_true}")
diff = np.abs(Pu - Pu_true)
print(f"Difference = \n{diff}")
print(f"Max difference = {np.max(diff)}")
