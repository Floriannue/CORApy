import numpy as np
from cora_python.contSet.polytope import Polytope

# Simulate the contractor scenario
A_ = np.array([[1, 2], [3, 4], [5, 6], [-1, -2], [-3, -4], [-5, -6]])
print('A_ shape:', A_.shape)

b_ = np.array([[1], [2], [3], [-0], [-1], [-2]])
print('b_ shape:', b_.shape)

n = 2
sup = np.array([10, 20])
infi = np.array([-10, -20])

A_full = np.vstack([A_, np.eye(n), -np.eye(n)])
b_full = np.vstack([b_, sup.reshape(-1, 1), -infi.reshape(-1, 1)])

print('A_full shape:', A_full.shape)
print('b_full shape:', b_full.shape)
print('A_full rows:', A_full.shape[0], 'b_full rows:', b_full.shape[0])

try:
    poly = Polytope(A_full, b_full)
    print('Success!')
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()

