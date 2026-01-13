"""
Compare remainder term computation and addition
Check if interval * zonotope conversion causes precision loss
"""
import numpy as np
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys.linearize import linearize

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

# Linearize
sys, linsys, linParams, linOptions = linearize(tank, params['R0'], params, options)

# Get inputs
U = linParams['U']
timeStep = options['timeStep']
truncationOrder = options['taylorTerms']

print("=" * 80)
print("COMPARING REMAINDER TERM ADDITION")
print("=" * 80)

# Decompose U
blocks = np.array([[1, linsys.nr_of_dims]])
U_decomp = U.decompose(blocks)

# Compute remainder term
from cora_python.contDynamics.linearSys.private.priv_expmRemainder import priv_expmRemainder
E = priv_expmRemainder(linsys, timeStep, truncationOrder)
E_times_dt = E * timeStep

print(f"\nE (remainder term):")
print(f"  Type: {type(E)}")
print(f"  Inf max: {np.max(E.inf)}")
print(f"  Sup max: {np.max(E.sup)}")

print(f"\nE * timeStep:")
print(f"  Type: {type(E_times_dt)}")
print(f"  Inf max: {np.max(E_times_dt.inf)}")
print(f"  Sup max: {np.max(E_times_dt.sup)}")

# Try direct multiplication (should fail for interval * zonotope)
print(f"\nTrying E*timeStep * U_decomp (direct):")
try:
    from cora_python.g.functions.helper.sets.contSet.contSet import block_mtimes
    result_direct = block_mtimes(E_times_dt, U_decomp)
    print(f"  [OK] Direct multiplication succeeded")
    print(f"  Result type: {type(result_direct)}")
    if hasattr(result_direct, 'c'):
        print(f"  Result center: {result_direct.c.flatten()}")
        print(f"  Result generators shape: {result_direct.G.shape}")
except Exception as e:
    print(f"  [FAILED] Direct multiplication failed: {e}")

# Convert U_decomp to intervals and multiply
print(f"\nConverting U_decomp to intervals and multiplying:")
from cora_python.contSet.interval import Interval
from cora_python.g.functions.helper.sets.contSet.contSet import block_operation

U_decomp_interval = block_operation(lambda x: Interval(x), U_decomp)
print(f"  U_decomp_interval type: {type(U_decomp_interval)}")
if isinstance(U_decomp_interval, Interval):
    print(f"  U_decomp_interval inf: {U_decomp_interval.inf.flatten()}")
    print(f"  U_decomp_interval sup: {U_decomp_interval.sup.flatten()}")

result_interval = block_mtimes(E_times_dt, U_decomp_interval)
print(f"  Result type: {type(result_interval)}")
if hasattr(result_interval, 'inf'):
    print(f"  Result inf: {result_interval.inf.flatten()}")
    print(f"  Result sup: {result_interval.sup.flatten()}")

# Compare with expected: the remainder term should add generators
# The issue might be in how interval * interval multiplication works
print(f"\nChecking interval * interval multiplication precision:")
# Manual computation: E_times_dt is (6,6) interval, U_decomp_interval is (6,1) interval
# Result should be (6,1) interval
if isinstance(E_times_dt, Interval) and isinstance(U_decomp_interval, Interval):
    from cora_python.contSet.interval.mtimes import mtimes
    result_manual = mtimes(E_times_dt, U_decomp_interval)
    print(f"  Manual result type: {type(result_manual)}")
    print(f"  Manual result inf: {result_manual.inf.flatten()}")
    print(f"  Manual result sup: {result_manual.sup.flatten()}")

print("\n" + "=" * 80)
