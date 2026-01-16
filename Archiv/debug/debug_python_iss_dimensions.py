"""Debug script to check dimensions in Python for ISS test case"""
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

print(f'Original system:')
print(f'  A shape: {A.shape}')
print(f'  B shape: {B.shape}')
print(f'  C shape: {C.shape}')

# Construct extended system
dim_x = A.shape[0]
A_top = np.hstack([A, B])
A_bottom = np.zeros((B.shape[1], dim_x + B.shape[1]))
A_ = np.vstack([A_top, A_bottom])
B_ = np.zeros((dim_x + B.shape[1], 1))
C_ = np.hstack([C, np.zeros((C.shape[0], B.shape[1]))])

print(f'\nExtended system:')
print(f'  A_ shape: {A_.shape}')
print(f'  B_ shape: {B_.shape}')
print(f'  C_ shape: {C_.shape}')

# Create system
sys = LinearSys('iss', A_, B_, None, C_)

print(f'\nLinearSys object:')
print(f'  sys.nr_of_dims: {sys.nr_of_dims}')
print(f'  sys.nr_of_inputs: {sys.nr_of_inputs}')
print(f'  sys.nr_of_outputs: {sys.nr_of_outputs}')

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
print(f'  params has V: {"V" in params}')
print(f'  sys.nr_of_outputs: {sys.nr_of_outputs}')

# Simulate aux_canonicalForm initialization
if 'V' not in params:
    params['V'] = Zonotope(np.zeros((sys.nr_of_outputs, 1)), 
                          np.zeros((sys.nr_of_outputs, 0)))

print(f'\nAfter V initialization:')
print(f'  params["V"].dim(): {params["V"].dim()}')
print(f'  sys.nr_of_outputs: {sys.nr_of_outputs}')

# Check Cs construction
from cora_python.contSet.polytope import Polytope
from cora_python.specification.specification import Specification

d = 5e-4
P1 = Polytope(np.array([[0, 0, 1]]), np.array([[-d]]))
P2 = Polytope(np.array([[0, 0, -1]]), np.array([[-d]]))
spec = Specification([P1, P2], 'unsafeSet')

# Simulate what happens in priv_verifyRA_supportFunc
# Get safe sets and unsafe sets (simplified)
unsafeSet = []
for i in range(len(spec)):
    if spec[i].type == 'unsafeSet':
        P = Polytope(spec[i].set)
        tmp = P.normalizeConstraints('A')
        unsafeSet.append({'set': tmp, 'time': spec[i].time})

nrSpecs = len(unsafeSet)
Cs = np.zeros((nrSpecs, sys.nr_of_outputs))

print(f'\nCs construction:')
print(f'  nrSpecs: {nrSpecs}')
print(f'  sys.nr_of_outputs: {sys.nr_of_outputs}')
print(f'  Cs shape: {Cs.shape}')
if len(unsafeSet) > 0:
    print(f'  unsafeSet[0]["set"].A shape: {unsafeSet[0]["set"].A.shape}')

Cs[0, :] = -unsafeSet[0]['set'].A
Cs[1, :] = -unsafeSet[1]['set'].A

print(f'\nAfter Cs assignment:')
print(f'  Cs shape: {Cs.shape}')
print(f'  Cs:\n{Cs}')

print(f'\nMatrix multiplication check:')
print(f'  Cs shape: {Cs.shape}')
print(f'  params["V"].generators().shape: {params["V"].generators().shape}')
print(f'  params["V"].dim(): {params["V"].dim()}')

try:
    result = Cs @ params['V'].generators()
    print(f'  Result shape: {result.shape}')
except Exception as e:
    print(f'  Error: {e}')
