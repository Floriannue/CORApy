import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.representsa_ import representsa_

print("=== Testing failing representsa_ cases ===\n")

# Test 1: 2D empty polytope (inequalities)
print("Test 1: 2D empty polytope (inequalities)")
A = np.array([[-1, -1], [-1, 1], [1, 0]])
b = np.array([-2, -2, -1])
P = Polytope(A, b)
print("Constraints:")
print("A:", A)
print("b:", b)
print("Polytope:", P)

# Check if it's actually empty by trying to find a feasible point
from scipy.optimize import linprog
try:
    # Try to find any feasible point
    c = np.zeros(2)
    res = linprog(c, A_ub=A, b_ub=b, bounds=None)
    print("Linprog result:", res)
    print("Success:", res.success)
    print("Status:", res.status)
    if res.success:
        print("Feasible point found:", res.x)
    else:
        print("No feasible point found (should be empty)")
except Exception as e:
    print("Linprog error:", e)

print("\nrepresentsa_(P, 'emptySet'):", representsa_(P, 'emptySet'))
print("-" * 50)

# Test 2: 2D empty polytope (equalities)
print("\nTest 2: 2D empty polytope (equalities)")
Ae = np.array([[1, 0], [0, 1], [1, 1]])
be = np.array([1, 1, 1])
P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
print("Constraints:")
print("Ae:", Ae)
print("be:", be)
print("Polytope:", P)

# Check if it's actually empty by trying to find a feasible point
try:
    # Try to find any feasible point
    c = np.zeros(2)
    res = linprog(c, A_eq=Ae, b_eq=be, bounds=None)
    print("Linprog result:", res)
    print("Success:", res.success)
    print("Status:", res.status)
    if res.success:
        print("Feasible point found:", res.x)
    else:
        print("No feasible point found (should be empty)")
except Exception as e:
    print("Linprog error:", e)

print("\nrepresentsa_(P, 'emptySet'):", representsa_(P, 'emptySet'))
print("-" * 50)

# Test our dual problem approach manually
print("\n=== Testing dual problem approach manually ===")

# For Test 1 (inequalities)
print("\nTest 1 dual problem:")
from cora_python.contSet.polytope.private.priv_equality_to_inequality import priv_equality_to_inequality

# Convert to dual problem: min b'*y s.t. A'*y = 0, y >= 0
A_aug, b_aug = priv_equality_to_inequality(A, b, np.zeros((0, 2)), np.zeros(0))
print("A_aug:", A_aug)
print("b_aug:", b_aug)

# Dual problem: min b'*y s.t. A'*y = 0, y >= 0
A_eq_dual = A_aug.T
b_eq_dual = np.zeros(2)
bounds_dual = [(0, None)] * len(b_aug)

try:
    res_dual = linprog(b_aug, A_eq=A_eq_dual, b_eq=b_eq_dual, bounds=bounds_dual)
    print("Dual problem result:", res_dual)
    print("Success:", res_dual.success)
    print("Objective value:", res_dual.fun)
    if res_dual.success:
        if res_dual.fun >= -1e-12:
            print("Result: NOT empty (b'*y >= 0)")
        else:
            print("Result: EMPTY (b'*y < 0)")
    else:
        print("Dual problem failed")
except Exception as e:
    print("Dual problem error:", e)
