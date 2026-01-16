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

# Add debug prints in expmOneParam to see what's happening
# For now, let's check what E_u_sum should be
print("Checking E_u_sum initialization...")
print("E_u should have shape based on g_u")
u = Zonotope(params['uTrans'], np.zeros((params['uTrans'].shape[0], 0)))
g_u = u.generators()
print("g_u shape:", g_u.shape, "g_u.size:", g_u.size)
