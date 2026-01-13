"""
Detailed step-by-step comparison of intermediate computations
"""
import numpy as np
import math
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.zonotope.center import center
from cora_python.contDynamics.nonlinearSys.linearize import linearize
from cora_python.contDynamics.contDynamics.linReach import linReach

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

# Get factors
options['factor'] = []
for i in range(1, options['taylorTerms'] + 2):
    options['factor'].append((options['timeStep'] ** i) / math.factorial(i))

print("=== STEP-BY-STEP INTERMEDIATE COMPUTATIONS ===\n")

# Step 1: Initial set
Rinit = {'set': params['R0'], 'error': np.zeros((dim_x, 1))}
print("Step 1: Initial Set")
print(f"  R0 center: {params['R0'].c.flatten()}")
print(f"  R0 generators shape: {params['R0'].G.shape}")
print(f"  R0 generators sum(abs): {np.sum(np.abs(params['R0'].G), axis=1)}")

# Step 2: Linearization
print("\nStep 2: Linearization")
sys, linsys, linParams, linOptions = linearize(tank, Rinit['set'], params, options)

cx = center(Rinit['set'])
print(f"  cx (center of R0): {cx.flatten()}")

f0prev = tank.mFile(cx, params['uTrans'])
print(f"  f0prev (dynamics at cx): {f0prev.flatten()}")

p_x = sys.linError.p.x
p_u = sys.linError.p.u
print(f"  p.x (linearization point): {p_x.flatten()}")
print(f"  p.u (linearization point): {p_u.flatten()}")

f0 = sys.linError.f0
print(f"  f0 (constant input): {f0.flatten()}")

A, B = tank.jacobian(p_x, p_u)
print(f"  A matrix (first row): {A[0, :]}")
print(f"  B matrix (first row): {B[0, :]}")

# Step 3: Check Taylor series computation
print("\nStep 3: Taylor Series Setup")
print(f"  timeStep: {options['timeStep']}")
print(f"  taylorTerms: {options['taylorTerms']}")
print(f"  factors: {options['factor']}")

# Step 4: Linearized system reachability
print("\nStep 4: Linearized System Reachability")
print(f"  linsys.A (first row): {linsys.A[0, :]}")
print(f"  linParams['U'] center: {center(linParams['U']).flatten()}")
print(f"  linParams['U'] generators shape: {linParams['U'].G.shape}")

# Step 5: linReach computation
print("\nStep 5: linReach Computation")
Rti, Rtp, dimForSplit, opts = linReach(tank, Rinit, params, options)

print(f"  Rti center: {Rti.c.flatten()}")
print(f"  Rti generators shape: {Rti.G.shape}")
print(f"  Rti generators sum(abs): {np.sum(np.abs(Rti.G), axis=1)}")

if isinstance(Rtp, dict):
    print(f"  Rtp['set'] center: {Rtp['set'].c.flatten()}")
    print(f"  Rtp['set'] generators shape: {Rtp['set'].G.shape}")
    print(f"  Rtp['set'] generators sum(abs): {np.sum(np.abs(Rtp['set'].G), axis=1)}")
    print(f"  Rtp['error']: {Rtp['error'].flatten()}")

# Step 6: Interval conversion
print("\nStep 6: Interval Conversion")
IH_tp = Interval(Rtp['set'] if isinstance(Rtp, dict) else Rtp)
print(f"  IH_tp.inf: {IH_tp.inf.flatten()[:6]}")
print(f"  IH_tp.sup: {IH_tp.sup.flatten()[:6]}")

# Expected values
IH_tp_true_inf = np.array([[1.8057949711597598], [3.6433030183959114], [3.7940260617482671], 
                          [1.9519553317477598], [9.3409949650858550], [4.0928655724716370]])
IH_tp_true_sup = np.array([[2.2288356782079028], [4.0572873081850807], [4.1960714210115002], 
                          [2.3451418924166987], [9.7630596270322201], [4.4862797486713282]])

print("\nStep 7: Comparison")
diff_inf = (IH_tp.inf.flatten()[:6] - IH_tp_true_inf.flatten())
diff_sup = (IH_tp.sup.flatten()[:6] - IH_tp_true_sup.flatten())
print(f"  Difference inf: {diff_inf}")
print(f"  Difference sup: {diff_sup}")
print(f"  Max abs diff inf: {np.max(np.abs(diff_inf))}")
print(f"  Max abs diff sup: {np.max(np.abs(diff_sup))}")
