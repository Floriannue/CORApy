"""
Test precision improvements with Kahan summation
"""
import numpy as np
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contDynamics.nonlinearSys.initReach import initReach

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
import math
options['factor'] = []
for i in range(1, options['taylorTerms'] + 2):
    options['factor'].append((options['timeStep'] ** i) / math.factorial(i))

print("=== TESTING PRECISION IMPROVEMENTS ===\n")

# Compute initReach with improved precision
Rfirst, _ = initReach(tank, params['R0'], params, options)

# Get interval hulls
IH_tp = Interval(Rfirst['tp'][0]['set'])
IH_ti = Interval(Rfirst['ti'][0])

print("Results with precision improvements:")
print(f"IH_tp inf: {IH_tp.inf.flatten()[:6]}")
print(f"IH_tp sup: {IH_tp.sup.flatten()[:6]}")

# Expected values
IH_tp_true_inf = np.array([[1.8057949711597598], [3.6433030183959114], [3.7940260617482671], 
                          [1.9519553317477598], [9.3409949650858550], [4.0928655724716370]])
IH_tp_true_sup = np.array([[2.2288356782079028], [4.0572873081850807], [4.1960714210115002], 
                          [2.3451418924166987], [9.7630596270322201], [4.4862797486713282]])

# Compare
diff_inf = (IH_tp.inf.flatten()[:6] - IH_tp_true_inf.flatten())
diff_sup = (IH_tp.sup.flatten()[:6] - IH_tp_true_sup.flatten())
print(f"\nDifference inf: {diff_inf}")
print(f"Difference sup: {diff_sup}")
print(f"Max abs diff inf: {np.max(np.abs(diff_inf))}")
print(f"Max abs diff sup: {np.max(np.abs(diff_sup))}")
