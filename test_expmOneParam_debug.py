from cora_python.matrixSet.matZonotope import matZonotope
from cora_python.matrixSet.matZonotope.expmOneParam import expmOneParam
from cora_python.contSet.zonotope import Zonotope
import numpy as np

C = np.array([[0, 1], [-1, -0.5]])
G = np.zeros((2, 2, 1))
G[:, :, 0] = np.array([[0.1, 0], [0, 0.1]])
matZ = matZonotope(C, G)

r = 0.1
maxOrder = 4
params = {
    'Uconst': Zonotope(np.array([[0], [0]]), np.array([[0.05, 0], [0, 0.05]])),
    'uTrans': np.array([[0.1], [0]])
}

try:
    eZ, eI, zPow, iPow, E, RconstInput = expmOneParam(matZ, r, maxOrder, params)
    print('SUCCESS')
    print('RconstInput.center():', RconstInput.center())
    print('MATLAB expected: [0.010008562500000; -0.000491329166667]')
    diff = RconstInput.center() - np.array([[0.010008562500000], [-0.000491329166667]])
    print('Difference:', diff.flatten())
    print('Max difference:', np.max(np.abs(diff)))
except Exception as e:
    print('ERROR:', e)
    import traceback
    traceback.print_exc()
