"""
Simple debug script to check dimensions in priv_reach_wrappingfree
Focus on the dimension mismatch issue
"""

import numpy as np
import sys
sys.path.insert(0, 'cora_python')

from contDynamics.linearSys.linearSys import LinearSys
from contSet.zonotope import Zonotope
from contSet.interval import Interval

# Create a simple 2D system
A = np.array([[-0.1, -2], [2, -0.1]])
B = np.array([[1], [0]])
sys = LinearSys(A, B)

# Parameters
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

# Use reach which handles canonicalForm internally
# This matches what the test does
from contDynamics.linearSys.reach import reach

# Add tStart to params
params['tStart'] = 0

# Call reach to get the internal values
# We'll trace through priv_reach_wrappingfree manually
from contDynamics.linearSys.private.priv_reach_wrappingfree import priv_reach_wrappingfree

# Call priv_reach_wrappingfree which does canonicalForm internally
# But first, let's manually trace what happens
# Put system into canonical form (this is what reach does)
sys_canon, U, u, V, v = sys.canonicalForm(
    params['U'], np.zeros((1, 1)),  # uTrans is zero for this test
    None, None,  # W and V are None
    np.zeros((sys.nr_of_inputs, 1))
)

# Now call oneStep with the canonical form
Rtp, Rti, Htp, Hti, PU, Pu, _, C_input = sys_canon.oneStep(
    params['R0'], U, u, options['timeStep'], options['taylorTerms'])

print('=== After oneStep ===')
print(f'PU type: {type(PU).__name__}')
if hasattr(PU, 'c'):
    print(f'PU center shape: {PU.c.shape}')
    if hasattr(PU, 'G') and PU.G.size > 0:
        print(f'PU generators shape: {PU.G.shape}')

# Read out propagation matrix
eAdt = sys.getTaylor('eAdt', {'timeStep': options['timeStep']})
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
print(f'PU.inf ndim: {PU.inf.ndim}')
print(f'PU.sup ndim: {PU.sup.ndim}')

# Check Pu
if hasattr(Pu, 'center'):
    Pu_c = Pu.center()
    print(f'\nPu type: {type(Pu).__name__}')
    print(f'Pu center shape: {Pu_c.shape}')
    print(f'Pu center ndim: {Pu_c.ndim}')
    Pu_interval = Pu.interval()
    print(f'Pu_interval.inf shape: {Pu_interval.inf.shape}')
    print(f'Pu_interval.inf ndim: {Pu_interval.inf.ndim}')
    Pu_c_1d = Pu_c.flatten() if Pu_c.ndim > 1 else Pu_c
    print(f'Pu_c_1d shape: {Pu_c_1d.shape}')
    print(f'Pu_c_1d ndim: {Pu_c_1d.ndim}')
    Pu_int = Pu_interval - Pu_c_1d
    print(f'Pu_int type: {type(Pu_int).__name__}')
    if hasattr(Pu_int, 'inf'):
        print(f'Pu_int.inf shape: {Pu_int.inf.shape}')
        print(f'Pu_int.inf ndim: {Pu_int.inf.ndim}')
        print(f'Pu_int.sup shape: {Pu_int.sup.shape}')
        print(f'Pu_int.sup ndim: {Pu_int.sup.ndim}')
    else:
        print(f'Pu_int shape: {Pu_int.shape}')
        print(f'Pu_int ndim: {Pu_int.ndim}')
else:
    Pu_c = Pu
    Pu_int = np.zeros((sys.nr_of_dims,))
    print(f'\nPu is numeric, shape: {Pu.shape}')

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
print(f'PU_next_interval.inf ndim: {PU_next_interval.inf.ndim}')
print(f'PU_next_interval.sup shape: {PU_next_interval.sup.shape}')
print(f'PU_next_interval.sup ndim: {PU_next_interval.sup.ndim}')

# Add intervals step by step
print('\n=== Adding intervals step by step ===')
PU_temp1 = PU + PU_next_interval
print(f'PU_temp1 (PU + PU_next_interval) type: {type(PU_temp1).__name__}')
print(f'PU_temp1.inf shape: {PU_temp1.inf.shape}')
print(f'PU_temp1.inf ndim: {PU_temp1.inf.ndim}')
print(f'PU_temp1.sup shape: {PU_temp1.sup.shape}')
print(f'PU_temp1.sup ndim: {PU_temp1.sup.ndim}')

# Ensure Pu_int is 1D
if isinstance(Pu_int, np.ndarray) and Pu_int.ndim > 1:
    Pu_int = Pu_int.flatten()
    print(f'\nPu_int flattened to shape: {Pu_int.shape}')

PU = PU_temp1 + Pu_int
print(f'\nPU (final after + Pu_int) type: {type(PU).__name__}')
print(f'PU.inf shape: {PU.inf.shape}')
print(f'PU.inf ndim: {PU.inf.ndim}')
print(f'PU.sup shape: {PU.sup.shape}')
print(f'PU.sup ndim: {PU.sup.ndim}')

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
print(f'PU.inf shape: {PU.inf.shape}')
print(f'PU.sup shape: {PU.sup.shape}')

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
