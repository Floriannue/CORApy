"""
Debug script to check dimensions in priv_reach_wrappingfree
Compare against MATLAB implementation
Uses the exact same setup as test_reach_different_algorithms
"""

import numpy as np
import sys
sys.path.insert(0, 'cora_python')

from contDynamics.linearSys.linearSys import LinearSys
from contSet.zonotope import Zonotope
from contSet.interval import Interval

# Use exact same setup as test_reach_different_algorithms
A = np.array([[-0.1, -2], [2, -0.1]])
B = np.array([[1], [0]])
sys = LinearSys(A, B)

# Parameters (exact same as test)
params = {
    'tFinal': 0.2,
    'R0': Zonotope(np.array([[10], [5]]), 0.5 * np.eye(2)),
    'U': Zonotope(np.zeros((1, 1)), 0.25 * np.eye(1))
}

# Options
options = {
    'timeStep': 0.05,
    'taylorTerms': 4,
    'zonotopeOrder': 20,
    'linAlg': 'wrapping-free'
}

# Call reach which handles canonicalForm internally
from contDynamics.linearSys.reach import reach
R = reach(sys, params, options)

# Now extract the internal values by calling oneStep directly
# This matches what priv_reach_wrappingfree does
from contDynamics.linearSys.private.priv_reach_wrappingfree import priv_reach_wrappingfree

# Recreate params with tStart
params_full = params.copy()
params_full['tStart'] = 0

# Call priv_reach_wrappingfree to see dimensions
# But first, let's manually trace through oneStep
Rtp, Rti, Htp, Hti, PU, Pu, _, C_input = sys.oneStep(
    params['R0'], params['U'], np.zeros((1, 1)), options['timeStep'], options['taylorTerms'])

print('=== After oneStep ===')
print(f'PU type: {type(PU).__name__}')
if hasattr(PU, 'c'):
    print(f'PU center shape: {PU.c.shape}')
    if hasattr(PU, 'G') and PU.G.size > 0:
        print(f'PU generators shape: {PU.G.shape}')

# Read out propagation matrix
eAdt = sys_canon.getTaylor('eAdt', {'timeStep': options['timeStep']})
print(f'\neAdt shape: {eAdt.shape}')

# Save particular solution
PU_next = PU
print(f'\nPU_next type: {type(PU_next).__name__}')
if hasattr(PU_next, 'c'):
    print(f'PU_next center shape: {PU_next.c.shape}')

# Convert PU to interval
PU = PU.interval() if hasattr(PU, 'interval') else Interval(PU)
print(f'\nPU (after interval conversion) type: {type(PU).__name__}')
print(f'PU.inf shape: {PU.inf.shape}')
print(f'PU.sup shape: {PU.sup.shape}')

# Check Pu
if hasattr(Pu, 'center'):
    Pu_c = Pu.center()
    print(f'\nPu type: {type(Pu).__name__}')
    print(f'Pu center shape: {Pu_c.shape}')
    Pu_interval = Pu.interval()
    Pu_c_1d = Pu_c.flatten() if Pu_c.ndim > 1 else Pu_c
    Pu_int = Pu_interval - Pu_c_1d
    print(f'Pu_int type: {type(Pu_int).__name__}')
    if hasattr(Pu_int, 'inf'):
        print(f'Pu_int.inf shape: {Pu_int.inf.shape}')
        print(f'Pu_int.sup shape: {Pu_int.sup.shape}')
    else:
        print(f'Pu_int shape: {Pu_int.shape}')
else:
    Pu_c = Pu
    Pu_int = np.zeros((sys.nr_of_dims,))
    print(f'\nPu is numeric, shape: {Pu.shape}')
    print(f'Pu_int shape: {Pu_int.shape}')

# Check C_input
print(f'\nC_input type: {type(C_input).__name__}')
if hasattr(C_input, 'c'):
    print(f'C_input center shape: {C_input.c.shape}')
    if hasattr(C_input, 'G') and C_input.G.size > 0:
        print(f'C_input generators shape: {C_input.G.shape}')

# Simulate one iteration
print('\n=== After one iteration ===')

# Propagate particular solution
PU_next = eAdt @ PU_next
print(f'PU_next (after eAdt multiplication) type: {type(PU_next).__name__}')
if hasattr(PU_next, 'c'):
    print(f'PU_next center shape: {PU_next.c.shape}')

# Convert to interval
PU_next_interval = PU_next.interval()
print(f'PU_next_interval type: {type(PU_next_interval).__name__}')
print(f'PU_next_interval.inf shape: {PU_next_interval.inf.shape}')
print(f'PU_next_interval.sup shape: {PU_next_interval.sup.shape}')

# Add intervals
PU_temp = PU + PU_next_interval
print(f'\nPU_temp (PU + PU_next_interval) type: {type(PU_temp).__name__}')
print(f'PU_temp.inf shape: {PU_temp.inf.shape}')
print(f'PU_temp.sup shape: {PU_temp.sup.shape}')

# Ensure Pu_int is 1D
if isinstance(Pu_int, np.ndarray) and Pu_int.ndim > 1:
    Pu_int = Pu_int.flatten()

PU = PU + PU_next_interval + Pu_int
print(f'\nPU (final) type: {type(PU).__name__}')
print(f'PU.inf shape: {PU.inf.shape}')
print(f'PU.sup shape: {PU.sup.shape}')

# Check Hti
print(f'\nHti type: {type(Hti).__name__}')
if hasattr(Hti, 'c'):
    print(f'Hti center shape: {Hti.c.shape}')
    if hasattr(Hti, 'G') and Hti.G.size > 0:
        print(f'Hti generators shape: {Hti.G.shape}')

# Try to add Hti + PU
print('\n=== Attempting Hti + PU ===')
print(f'Hti dimension: {Hti.dim()}')
print(f'PU dimension: {PU.dim()}')

try:
    Rti_test = Hti + PU
    print('SUCCESS: Hti + PU works')
    print(f'Rti_test type: {type(Rti_test).__name__}')
    if hasattr(Rti_test, 'c'):
        print(f'Rti_test center shape: {Rti_test.c.shape}')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {str(e)}')
    import traceback
    traceback.print_exc()
