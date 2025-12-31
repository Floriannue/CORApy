"""Test convergence check with point interval"""
import numpy as np
from cora_python.contSet.interval.interval import Interval

# Simulate the convergence check after first iteration
dom_original = Interval([1, 1], [3, 3])
dom_after_iter1 = Interval([1.0981, 1.0981], [1.0981, 1.0981])

print("=== Testing convergence check ===")
print(f"dom_original: {dom_original}")
print(f"dom_after_iter1: {dom_after_iter1}")

# Check if domain is a point (width is zero)
width = dom_after_iter1.sup - dom_after_iter1.inf
print(f"\nWidth: {width}")
eps = np.finfo(float).eps
is_point = np.all(np.abs(width) < eps)
print(f"Is point interval: {is_point}")
print(f"eps: {eps}")

# Current convergence check
diff_inf = np.abs(dom_after_iter1.inf - dom_original.inf)
diff_sup = np.abs(dom_after_iter1.sup - dom_original.sup)
converged_old = np.all(diff_inf < eps) and np.all(diff_sup < eps)
print(f"\nOld convergence check: {converged_old}")

# New convergence check (with point check)
converged_new = is_point or (np.all(diff_inf < eps) and np.all(diff_sup < eps))
print(f"New convergence check (with point): {converged_new}")

# Check if point satisfies constraint
def f(x):
    return x[0]**2 + x[1]**2 - 4

point_val = f([1.0981, 1.0981])
print(f"\nf(point): {point_val}")
print(f"Point satisfies f(x)=0: {abs(point_val) < 1e-10}")

