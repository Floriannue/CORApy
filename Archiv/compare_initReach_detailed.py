"""
Detailed comparison script for initReach - outputs all intermediate values
"""
import numpy as np
import math
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contDynamics.nonlinearSys.initReach import initReach
from cora_python.contDynamics.contDynamics.linReach import linReach
from cora_python.contDynamics.nonlinearSys.linearize import linearize
from cora_python.contSet.zonotope.center import center

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

print("=== DETAILED COMPARISON OUTPUT ===")
print(f"R0 center: {params['R0'].c.flatten()}")
print(f"R0 generators shape: {params['R0'].G.shape}")
print(f"U center: {params['U'].c.flatten()}")
print(f"U generators shape: {params['U'].G.shape}")
print(f"timeStep: {options['timeStep']}")
print(f"taylorTerms: {options['taylorTerms']}")
print(f"factors: {options['factor']}")

# Step 1: Linearization
Rinit = {'set': params['R0'], 'error': np.zeros((dim_x, 1))}
sys, linsys, linParams, linOptions = linearize(tank, Rinit['set'], params, options)

print(f"\n--- Linearization ---")
# linError.p is an object with x and u attributes
if hasattr(sys.linError.p, 'x'):
    p_x = sys.linError.p.x
    p_u = sys.linError.p.u
else:
    p_x = None
    p_u = None
print(f"p.x (linearization point): {p_x.flatten() if p_x is not None else 'N/A'}")
print(f"p.u (linearization point): {p_u.flatten() if p_u is not None else 'N/A'}")
print(f"f0 (constant input): {sys.linError.f0.flatten()}")
if hasattr(linsys, 'A'):
    print(f"A matrix:\n{linsys.A}")
if hasattr(linsys, 'B'):
    print(f"B matrix:\n{linsys.B}")

# Step 2: linReach
Rti, Rtp, dimForSplit, opts = linReach(tank, Rinit, params, options)

print(f"\n--- linReach Results ---")
print(f"Rti center: {Rti.c.flatten()}")
print(f"Rti generators shape: {Rti.G.shape}")
if isinstance(Rtp, dict):
    print(f"Rtp['set'] center: {Rtp['set'].c.flatten()}")
    print(f"Rtp['set'] generators shape: {Rtp['set'].G.shape}")
    print(f"Rtp['error']: {Rtp['error'].flatten()}")

# Step 3: Interval hulls
IH_ti = Interval(Rti)
IH_tp = Interval(Rtp['set'] if isinstance(Rtp, dict) else Rtp)

print(f"\n--- Interval Hulls ---")
# Ensure inf and sup are column vectors
IH_ti_inf = IH_ti.inf.flatten() if IH_ti.inf.ndim > 1 else IH_ti.inf
IH_ti_sup = IH_ti.sup.flatten() if IH_ti.sup.ndim > 1 else IH_ti.sup
IH_tp_inf = IH_tp.inf.flatten() if IH_tp.inf.ndim > 1 else IH_tp.inf
IH_tp_sup = IH_tp.sup.flatten() if IH_tp.sup.ndim > 1 else IH_tp.sup

print(f"IH_ti inf shape: {IH_ti.inf.shape}, value: {IH_ti_inf[:6]}")
print(f"IH_ti sup shape: {IH_ti.sup.shape}, value: {IH_ti_sup[:6]}")
print(f"IH_tp inf shape: {IH_tp.inf.shape}, value: {IH_tp_inf[:6]}")
print(f"IH_tp sup shape: {IH_tp.sup.shape}, value: {IH_tp_sup[:6]}")

# Expected values
IH_tp_true_inf = np.array([[1.8057949711597598], [3.6433030183959114], [3.7940260617482671], 
                          [1.9519553317477598], [9.3409949650858550], [4.0928655724716370]])
IH_tp_true_sup = np.array([[2.2288356782079028], [4.0572873081850807], [4.1960714210115002], 
                          [2.3451418924166987], [9.7630596270322201], [4.4862797486713282]])

print(f"\n--- Comparison with Expected ---")
# Ensure shapes match
IH_tp_inf_flat = IH_tp_inf[:6] if len(IH_tp_inf) >= 6 else IH_tp_inf
IH_tp_sup_flat = IH_tp_sup[:6] if len(IH_tp_sup) >= 6 else IH_tp_sup
IH_tp_true_inf_flat = IH_tp_true_inf.flatten()
IH_tp_true_sup_flat = IH_tp_true_sup.flatten()

diff_inf = IH_tp_inf_flat - IH_tp_true_inf_flat
diff_sup = IH_tp_sup_flat - IH_tp_true_sup_flat
print(f"Difference IH_tp inf: {diff_inf}")
print(f"Difference IH_tp sup: {diff_sup}")
print(f"Relative error IH_tp inf: {(diff_inf / np.abs(IH_tp_true_inf_flat))}")
print(f"Relative error IH_tp sup: {(diff_sup / np.abs(IH_tp_true_sup_flat))}")
print(f"\nMax absolute difference inf: {np.max(np.abs(diff_inf))}")
print(f"Max absolute difference sup: {np.max(np.abs(diff_sup))}")
print(f"Max relative error inf: {np.max(np.abs(diff_inf / np.abs(IH_tp_true_inf_flat)))}")
print(f"Max relative error sup: {np.max(np.abs(diff_sup / np.abs(IH_tp_true_sup_flat)))}")
