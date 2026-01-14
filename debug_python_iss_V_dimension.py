"""Debug script to check params['V'] dimension issue"""
import scipy.io
import os
import numpy as np
from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.interval.vertcat import vertcat

# Load system matrices
mat_file_path = os.path.join(CORAROOT(), 'models', 'Cora', 'iss.mat')
data = scipy.io.loadmat(mat_file_path)
A = data['A']
B = data['B']
C = data['C']

# Convert to dense
A = A.toarray() if hasattr(A, 'toarray') else A
B = B.toarray() if hasattr(B, 'toarray') else B
C = C.toarray() if hasattr(C, 'toarray') else C

# Construct extended system
dim_x = A.shape[0]
A_top = np.hstack([A, B])
A_bottom = np.zeros((B.shape[1], dim_x + B.shape[1]))
A_ = np.vstack([A_top, A_bottom])
B_ = np.zeros((dim_x + B.shape[1], 1))
C_ = np.hstack([C, np.zeros((C.shape[0], B.shape[1]))])

# Create system
sys = LinearSys('iss', A_, B_, None, C_)

print(f'sys.nr_of_outputs: {sys.nr_of_outputs}')
print(f'sys.D: {sys.D}')
print(f'sys.D is None: {sys.D is None}')
print(f'sys.D shape (if not None): {sys.D.shape if sys.D is not None else "N/A"}')

# Create params
params = {
    'U': Zonotope(np.array([[0]]), np.zeros((1, 0))),
    'R0': vertcat(
        Interval(-0.0001 * np.ones((270, 1)), 0.0001 * np.ones((270, 1))),
        Interval(np.array([[0], [0.8], [0.9]]), np.array([[0.1], [1], [1]]))
    ).zonotope(),
    'tFinal': 20
}

print(f'\nBefore aux_canonicalForm:')
print(f'  params["U"].dim(): {params["U"].dim()}')

# Simulate aux_canonicalForm
U_input_space = params['U']
if 'V' not in params:
    params['V'] = Zonotope(np.zeros((sys.nr_of_outputs, 1)), 
                          np.zeros((sys.nr_of_outputs, 0)))

print(f'\nAfter V initialization:')
print(f'  params["V"].dim(): {params["V"].dim()}')

print(f'\nChecking D condition:')
print(f'  sys.D is not None: {sys.D is not None}')
print(f'  np.any(sys.D): {np.any(sys.D)}')
print(f'  Condition result: {sys.D is not None and np.any(sys.D)}')

if sys.D is not None and np.any(sys.D):
    print(f'\nD is not None and has non-zero values')
    print(f'  sys.D shape: {sys.D.shape}')
    print(f'  U_input_space.dim(): {U_input_space.dim()}')
    print(f'  U_input_space.generators().shape: {U_input_space.generators().shape}')
    print(f'  sys.D @ U_input_space would have shape: ({sys.D.shape[0]}, {U_input_space.generators().shape[1]})')
    result = sys.D @ U_input_space
    print(f'  sys.D @ U_input_space result type: {type(result)}')
    print(f'  sys.D @ U_input_space result dim: {result.dim() if hasattr(result, "dim") else "N/A"}')
    print(f'  params["V"].dim() before: {params["V"].dim()}')
    params['V'] = sys.D @ U_input_space + params['V']
    print(f'  After D @ U + V, params["V"].dim(): {params["V"].dim()}')
else:
    print(f'\nD condition is False, V not modified')

print(f'\nFinal params["V"].dim(): {params["V"].dim()}')
