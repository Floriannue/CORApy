import numpy as np
from cora_python.contSet.polytope.polytope import Polytope

# Test the point case that's failing
A = np.array([[1]])
b = np.array([1])
P1 = Polytope(A, b)

p = np.array([[5]])
result = P1.isIntersecting_(p, 'exact')
print("Result type:", type(result))
print("Result value:", result)
print("Result repr:", repr(result)) 