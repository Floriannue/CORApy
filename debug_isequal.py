import numpy as np
import sympy as sp
from cora_python.hybridDynamics.linearReset import LinearReset
from cora_python.hybridDynamics.nonlinearReset import NonlinearReset
from cora_python.g.functions.matlab.function_handle.inputArgsLength import inputArgsLength

# Create the actual test case
A = np.array([[1, 0], [0, 1]])
linReset = LinearReset(A)
nonlinReset = linReset.nonlinearReset()

def f(x, u):
    return np.array([[x[0]], [x[1]]])
nonlinReset_ = NonlinearReset(f)

# Get input dimensions
f_in, f_out = inputArgsLength(nonlinReset.f, 2)
g_in, g_out = inputArgsLength(nonlinReset_.f, 2)
print(f'f_in: {f_in}, f_out: {f_out}')
print(f'g_in: {g_in}, g_out: {g_out}')

# Create symbolic variables
argsIn = []
for i in range(len(f_in)):
    varName = f'argsIn_{i}_'
    if f_in[i] == 1:
        argsIn.append(np.array([sp.Symbol(f'{varName}1', real=True)]))
    else:
        argsIn.append(np.array([sp.Symbol(f'{varName}{j+1}', real=True) for j in range(f_in[i])]))

# Evaluate functions
f_sym = nonlinReset.f(*argsIn)
g_sym = nonlinReset_.f(*argsIn)

print(f'\nf_sym type: {type(f_sym)}')
print(f'f_sym: {f_sym}')
print(f'f_sym shape: {f_sym.shape if hasattr(f_sym, "shape") else "N/A"}')

print(f'\ng_sym type: {type(g_sym)}')
print(f'g_sym: {g_sym}')
print(f'g_sym shape: {g_sym.shape if hasattr(g_sym, "shape") else "N/A"}')

# Convert to sympy matrices
if isinstance(f_sym, np.ndarray):
    try:
        f_sym_mat = sp.Matrix(f_sym.tolist())
        print(f'\nf_sym_mat: {f_sym_mat}')
    except Exception as e:
        print(f'Error converting f_sym: {e}')
        f_sym_mat = None
else:
    f_sym_mat = f_sym

if isinstance(g_sym, np.ndarray):
    try:
        g_sym_mat = sp.Matrix(g_sym.tolist())
        print(f'g_sym_mat: {g_sym_mat}')
    except Exception as e:
        print(f'Error converting g_sym: {e}')
        g_sym_mat = None
else:
    g_sym_mat = g_sym

# Compare
if f_sym_mat is not None and g_sym_mat is not None:
    diff = sp.simplify(f_sym_mat - g_sym_mat)
    print(f'\ndiff: {diff}')
    print(f'diff simplified: {sp.simplify(diff)}')
    for i in range(diff.rows):
        for j in range(diff.cols):
            elem = sp.simplify(diff[i, j])
            print(f'diff[{i},{j}]: {elem}, == 0: {elem == 0}, is_zero: {elem.is_zero if hasattr(elem, "is_zero") else "N/A"}')

