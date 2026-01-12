import numpy as np
import sympy as sp
from cora_python.hybridDynamics.linearReset import LinearReset
from cora_python.hybridDynamics.nonlinearReset import NonlinearReset
from cora_python.g.functions.matlab.function_handle.isequalFunctionHandle import isequalFunctionHandle

# Create linear reset and convert
A = np.array([[1, 0], [0, 1]])
linReset = LinearReset(A)
nonlinReset = linReset.nonlinearReset()

# Create direct function
def f(x, u):
    return np.array([[x[0]], [x[1]]])

nonlinReset_ = NonlinearReset(f)

print(f'nonlinReset.f: {nonlinReset.f}')
print(f'nonlinReset_.f: {nonlinReset_.f}')

# Test isequalFunctionHandle directly
try:
    result = isequalFunctionHandle(nonlinReset.f, nonlinReset_.f)
    print(f'isequalFunctionHandle result: {result}')
except Exception as e:
    print(f'Error in isequalFunctionHandle: {e}')
    import traceback
    traceback.print_exc()

# Test isequal
try:
    result = nonlinReset.isequal(nonlinReset_)
    print(f'isequal result: {result}')
except Exception as e:
    print(f'Error in isequal: {e}')
    import traceback
    traceback.print_exc()

