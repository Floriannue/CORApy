"""Test convergence check logic"""
import numpy as np
from cora_python.contSet.interval.interval import Interval

# Simulate the convergence check
dom_original = Interval([1, 1], [3, 3])
dom_after_iter1 = Interval([1.0981, 1.0981], [1.0981, 1.0981])

print("=== Testing convergence check ===")
print(f"dom_original: {dom_original}")
print(f"dom_after_iter1: {dom_after_iter1}")

# Check convergence (comparing with original)
diff_inf = np.abs(dom_after_iter1.inf - dom_original.inf)
diff_sup = np.abs(dom_after_iter1.sup - dom_original.sup)
eps = np.finfo(float).eps

print(f"\ndiff_inf: {diff_inf}")
print(f"diff_sup: {diff_sup}")
print(f"eps: {eps}")

converged = np.all(diff_inf < eps) and np.all(diff_sup < eps)
print(f"Converged (comparing with original): {converged}")

# Check if domain is a point (width is zero)
width = dom_after_iter1.sup - dom_after_iter1.inf
print(f"\nWidth: {width}")
is_point = np.all(np.abs(width) < eps)
print(f"Is point interval: {is_point}")

# If it's a point, we should consider it converged
if is_point:
    print("Since it's a point interval, we should consider it converged!")

