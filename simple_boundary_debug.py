import sys
sys.path.insert(0, 'cora_python')
import numpy as np

# Simple boundary test
inf = np.array([-2., -1.])
sup = np.array([3., 4.])
point = np.array([3., 4.])
tol = 1e-12

print(f"inf: {inf}")
print(f"sup: {sup}")
print(f"point: {point}")
print(f"tol: {tol}")

# Check basic conditions
print(f"inf <= point: {inf <= point}")
print(f"point <= sup: {point <= sup}")

# Check with tolerance
print(f"inf < point + tol: {inf < point + tol}")
print(f"sup > point - tol: {sup > point - tol}")

# The problem case
print(f"sup > point - tol detail:")
print(f"  sup: {sup}")
print(f"  point - tol: {point - tol}")
print(f"  sup > point - tol: {sup > point - tol}")

# For the boundary point [3, 4], sup is [3, 4] and point-tol is [3-1e-12, 4-1e-12]
# So we need [3, 4] > [3-1e-12, 4-1e-12] which should be True
# But let's check the exact values
print(f"Exact check:")
print(f"  3.0 > 3.0 - 1e-12 = 3.0 > {3.0 - 1e-12} = {3.0 > 3.0 - 1e-12}")
print(f"  4.0 > 4.0 - 1e-12 = 4.0 > {4.0 - 1e-12} = {4.0 > 4.0 - 1e-12}")

# The issue might be that we need >= instead of > for boundary cases
print(f"With >= for boundary:")
print(f"  inf <= point: {inf <= point}")  
print(f"  point <= sup: {point <= sup}")

# Check if all conditions are met
all_conditions = np.all(inf <= point) and np.all(point <= sup)
print(f"All boundary conditions met: {all_conditions}") 