import numpy as np
from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset

# Test the problematic function
def f(x, u):
    return np.array([[x[0]], [x[1]]])  # Column vector

try:
    nonlinReset = NonlinearReset(f)
    print(f'Success! preStateDim: {nonlinReset.preStateDim}, inputDim: {nonlinReset.inputDim}, postStateDim: {nonlinReset.postStateDim}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

