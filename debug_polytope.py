import numpy as np
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.polytope.private import priv_compact_all, priv_equality_to_inequality, priv_normalize_constraints

# Test case from the failing test
A_outside = np.array([[1, 0], [-1, 1], [-1, -1]])
b_outside = np.array([[7], [-2], [-4]])

print('Original constraints:')
print('A:', A_outside)
print('b:', b_outside)

# Step 1: Convert equality to inequality
A, b = priv_equality_to_inequality(A_outside, b_outside, np.array([]).reshape(0,2), np.array([]).reshape(0,1))
print('\nAfter equality to inequality:')
print('A:', A)
print('b:', b)

# Step 2: Normalize constraints
A, b, _, _ = priv_normalize_constraints(A, b, np.array([]).reshape(0,2), np.array([]).reshape(0,1), 'A')
print('\nAfter normalization:')
print('A:', A)
print('b:', b)

# Step 3: Compact all
A, b, _, _, empty, _ = priv_compact_all(A, b, np.array([]).reshape(0,2), np.array([]).reshape(0,1), 2, 1e-12)
print('\nAfter compact_all:')
print('A:', A)
print('b:', b)
print('empty:', empty)

# Step 4: Check if the polytope is actually empty by solving a simple LP
if A.size > 0:
    from scipy.optimize import linprog
    # Try to find a feasible point
    c = np.zeros(2)  # objective doesn't matter
    result = linprog(c, A_ub=A, b_ub=b, method='highs')
    print('\nLP result:')
    print('success:', result.success)
    print('x:', result.x)
    print('message:', result.message)
else:
    print('\nNo constraints left after compaction') 