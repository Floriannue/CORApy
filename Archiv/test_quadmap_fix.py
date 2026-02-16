"""test_quadmap_fix - Test if the quadMap fix is working"""

import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval

print("=" * 80)
print("TESTING quadMap FIX")
print("=" * 80)

# Create a simple zonotope
Z = Zonotope(np.array([[1.0], [0.0]]), np.array([[1.0, 0.5], [0.0, 0.3]]))

# Create an Interval Hessian
H_interval = Interval(
    np.array([[-0.1, -0.05], [-0.05, -0.1]]),  # inf
    np.array([[0.1, 0.05], [0.05, 0.1]])       # sup
)

H = [H_interval, H_interval]

print(f"\nH[0] inf max: {np.max(np.abs(H_interval.inf))}")
print(f"H[0] sup max: {np.max(np.abs(H_interval.sup))}")
print(f"H[0] center max: {np.max(np.abs(H_interval.center()))}")

# Test with old method (center)
print("\n1. Old method (center):")
Zmat = np.hstack([Z.c, Z.G])
quadMat_old = Zmat.T @ H[0] @ Zmat
if isinstance(quadMat_old, Interval):
    quadMat_old_center = quadMat_old.center()
    print(f"  quadMat center diagonal: {np.diag(quadMat_old_center[1:3, 1:3])}")
    print(f"  Using center: {np.diag(quadMat_old_center[1:3, 1:3])}")

# Test with new method (max(abs(inf), abs(sup)))
print("\n2. New method (max(abs(inf), abs(sup))):")
quadMat_new = Zmat.T @ H[0] @ Zmat
if isinstance(quadMat_new, Interval):
    quadMat_new_inf = quadMat_new.inf
    quadMat_new_sup = quadMat_new.sup
    quadMat_new_max = np.maximum(np.abs(quadMat_new_inf), np.abs(quadMat_new_sup))
    print(f"  quadMat inf diagonal: {np.diag(quadMat_new_inf[1:3, 1:3])}")
    print(f"  quadMat sup diagonal: {np.diag(quadMat_new_sup[1:3, 1:3])}")
    print(f"  max(abs(inf), abs(sup)) diagonal: {np.diag(quadMat_new_max[1:3, 1:3])}")

# Test actual quadMap call
print("\n3. Actual quadMap call:")
try:
    errorSec = 0.5 * Z.quadMap(H)
    print(f"  errorSec radius max: {np.max(np.sum(np.abs(errorSec.generators()), axis=1))}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print("Old method (center): Uses midpoint - less conservative")
print("New method (max(abs)): Uses maximum absolute - more conservative")
print("MATLAB likely uses the new method (max(abs)) for consistency")
print("=" * 80)
