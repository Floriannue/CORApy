"""
Detailed debug script to trace affineSolution computation step-by-step
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

# Initialize taylor if needed
if not hasattr(sys, 'taylor') or sys.taylor is None:
    from cora_python.g.classes.taylorLinSys import TaylorLinSys
    sys.taylor = TaylorLinSys(sys.A)

center = np.array([[40], [20]], dtype=float)
generators = np.array([[1, 4, 2], [-1, 3, 5]], dtype=float)
X = Zonotope(center, generators)

u = np.array([[2], [-1]], dtype=float)
timeStep = 0.05
truncationOrder = 6

print("=== Step-by-step computation ===\n")

# Check if Ainv exists
print(f"Ainv exists: {sys.taylor.Ainv is not None}")
if sys.taylor.Ainv is not None:
    print(f"Ainv = \n{sys.taylor.Ainv}")
    print(f"Ainv type: {type(sys.taylor.Ainv)}")

# Get eAdt
eAdt = sys.taylor.getTaylor('eAdt', timeStep=timeStep)
print(f"\neAdt = \n{eAdt}")
print(f"eAdt type: {type(eAdt)}")

# Check analytical computation
Ainv_analytical = np.linalg.inv(A)
eAdt_analytical = scipy.linalg.expm(A * timeStep)
print(f"\nAinv (analytical) = \n{Ainv_analytical}")
print(f"eAdt (analytical) = \n{eAdt_analytical}")

# Compare Ainv
if sys.taylor.Ainv is not None:
    diff_Ainv = np.abs(sys.taylor.Ainv - Ainv_analytical)
    print(f"\nMax difference in Ainv: {np.max(diff_Ainv)}")

# Compare eAdt
diff_eAdt = np.abs(eAdt - eAdt_analytical)
print(f"Max difference in eAdt: {np.max(diff_eAdt)}")

# Compute (eAdt - eye)
eye = np.eye(2)
eAdt_minus_eye = eAdt - eye
print(f"\neAdt - eye = \n{eAdt_minus_eye}")

# Compute Ainv * (eAdt - eye)
if sys.taylor.Ainv is not None:
    Ainv_eAdt_minus_eye = sys.taylor.Ainv @ (eAdt - eye)
    print(f"\nAinv * (eAdt - eye) = \n{Ainv_eAdt_minus_eye}")
    
    # Compare with analytical
    Ainv_eAdt_minus_eye_analytical = Ainv_analytical @ (eAdt_analytical - eye)
    print(f"Ainv * (eAdt - eye) (analytical) = \n{Ainv_eAdt_minus_eye_analytical}")
    diff_matrix = np.abs(Ainv_eAdt_minus_eye - Ainv_eAdt_minus_eye_analytical)
    print(f"Max difference in matrix: {np.max(diff_matrix)}")
    
    # Compute Pu
    Pu_computed = Ainv_eAdt_minus_eye @ u
    print(f"\nPu (computed step-by-step) = \n{Pu_computed}")
    
    # Analytical Pu
    Pu_analytical = Ainv_analytical @ (eAdt_analytical - eye) @ u
    print(f"Pu (analytical) = \n{Pu_analytical}")
    diff_Pu = np.abs(Pu_computed - Pu_analytical)
    print(f"Max difference in Pu: {np.max(diff_Pu)}")

# Now call affineSolution
print("\n=== Calling affineSolution ===")
Htp, Pu, Hti, C_state, C_input = affineSolution(sys, X, u, timeStep, truncationOrder)
print(f"Pu from affineSolution = \n{Pu}")
print(f"Pu type: {type(Pu)}")

# Compare
Pu_true = np.linalg.inv(A) @ (scipy.linalg.expm(A * timeStep) - np.eye(2)) @ u
print(f"\nPu_true = \n{Pu_true}")
diff = np.abs(Pu - Pu_true)
print(f"Max difference: {np.max(diff)}")
print(f"Difference: {diff}")
