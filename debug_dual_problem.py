import numpy as np
from scipy.optimize import linprog
from cora_python.contSet.polytope.private.priv_equality_to_inequality import priv_equality_to_inequality

print("=== Debugging dual problem approach ===\n")

# Test case from failing test
A = np.array([[-1, -1], [-1, 1], [1, 0]])
b = np.array([-2, -2, -1])
Ae = np.zeros((0, 2))
be = np.zeros(0)

print("Original constraints:")
print("A:", A)
print("b:", b)
print("Ae:", Ae)
print("be:", be)

# Convert to dual problem
A_aug, b_aug = priv_equality_to_inequality(A, b, Ae, be)
print("\nAfter priv_equality_to_inequality:")
print("A_aug:", A_aug)
print("b_aug:", b_aug)

# Set up the dual problem: min b'*y s.t. A'*y = 0, y >= 0
n = 2  # dimension
nrConIneq = len(b_aug)

print(f"\nDual problem setup:")
print(f"n = {n}, nrConIneq = {nrConIneq}")

# Objective: minimize b'*y
c_dual = b_aug.flatten()
print("c_dual (objective):", c_dual)

# Constraints: A'*y = 0 (equality) and y >= 0 (bounds)
A_eq_dual = A_aug.T
b_eq_dual = np.zeros(n)
bounds_dual = [(0, None)] * nrConIneq  # y >= 0

print("A_eq_dual:", A_eq_dual)
print("b_eq_dual:", b_eq_dual)
print("bounds_dual:", bounds_dual)

# Solve the dual problem
try:
    res_lp = linprog(c_dual, A_eq=A_eq_dual, b_eq=b_eq_dual, bounds=bounds_dual)
    print("\nDual problem result:")
    print("Success:", res_lp.success)
    print("Status:", res_lp.status)
    print("Objective value:", res_lp.fun)
    print("Solution y:", res_lp.x)
    
    if res_lp.success:
        if res_lp.fun >= -1e-12:
            print("Result: NOT empty (b'*y >= 0)")
        else:
            print("Result: EMPTY (b'*y < 0)")
    else:
        print("Dual problem failed")
        
except Exception as e:
    print("Dual problem error:", e)

# Let's also test the primal problem directly to see if it's actually infeasible
print("\n=== Testing primal problem directly ===")
try:
    # Try to find any feasible point
    c_primal = np.zeros(2)
    res_primal = linprog(c_primal, A_ub=A, b_ub=b, bounds=None)
    print("Primal problem result:")
    print("Success:", res_primal.success)
    print("Status:", res_primal.status)
    if res_primal.success:
        print("Feasible point found:", res_primal.x)
    else:
        print("No feasible point found (should be empty)")
except Exception as e:
    print("Primal problem error:", e)

# Let's also check what MATLAB's approach would do
print("\n=== MATLAB approach analysis ===")
print("The MATLAB approach converts equality constraints to inequalities:")
print("Each equality Ae*x = be becomes:")
print("  Ae*x <= be AND -Ae*x <= -be")

print("\nFor our case (no equality constraints):")
print("A_aug should be the same as A")
print("b_aug should be the same as b")

print("\nThe dual problem min b'*y s.t. A'*y = 0, y >= 0")
print("This checks if there exists a non-negative y such that:")
print("  A'*y = 0 (linear combination of constraints equals zero)")
print("  b'*y < 0 (negative objective)")

print("\nIf such a y exists, it means the polytope is empty.")
print("If no such y exists, the polytope is not empty.")
