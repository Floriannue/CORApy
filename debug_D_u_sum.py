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

# Manually trace D_u computation
c_u = params['uTrans']
print('c_u:', c_u.flatten())
print('D_u(:,:,1) should be c_u =', c_u.flatten())
print('D_u_sum starts with D_u(:,:,1)*r =', (c_u * r).flatten())

# Check what D_u should be after loop
print('\nAfter loop, D_u_sum should be:')
print('D_u(:,:,1)*r + sum(D_u(:,:,i)*r^i/factorial(i) for i=2:maxOrder)')
print('+ sum(0.5*E_u_sum(:,:,2*n) for n=1:floor(maxOrder/2))')
