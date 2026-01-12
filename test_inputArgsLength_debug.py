import numpy as np
import sympy as sp
from cora_python.g.functions.matlab.function_handle.inputArgsLength import inputArgsLength

# Test the problematic function
f = lambda x, u: np.array([[x[0]], [x[1]]])

try:
    count, out = inputArgsLength(f, 2)
    print(f'Success! count: {count}, out: {out}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

