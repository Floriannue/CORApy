import numpy as np
import sys
sys.path.append('cora_python')

from cora_python.contSet.interval.interval import Interval

# Recreate the test case
lb = np.zeros((2, 2, 1, 3, 2))
lb[:, :, 0, 0, 0] = [[1, 2], [3, 5]]
lb[:, :, 0, 1, 0] = [[0, -1], [-2, 3]]
lb[:, :, 0, 2, 0] = [[1, 1], [-1, 0]]
lb[:, :, 0, 0, 1] = [[-3, 2], [0, 1]]

ub = np.zeros((2, 2, 1, 3, 2))
ub[:, :, 0, 0, 0] = [[1.5, 4], [4, 10]]
ub[:, :, 0, 1, 0] = [[1, 2], [0, 4]]
ub[:, :, 0, 2, 0] = [[2, 3], [-0.5, 2]]
ub[:, :, 0, 0, 1] = [[-1, 3], [0, 2]]

I = Interval(lb, ub)
c = (lb + ub) / 2

print("Interval bounds shape:", lb.shape)
print("Interval inf flattened shape:", I.inf.flatten().shape)
print("Center shape:", c.shape)

# Test with center points
test_points = np.concatenate([c, c], axis=4)
print("Test points shape:", test_points.shape)
print("Test points flattened shape:", test_points.flatten().shape)

print("\nInterval inf (first few):", I.inf.flatten()[:10])
print("Interval sup (first few):", I.sup.flatten()[:10])
print("Center flattened (first few):", c.flatten()[:10])
print("Test points flattened (first few):", test_points.flatten()[:10])

# Check if center points are within bounds manually
c_flat = c.flatten()
inf_flat = I.inf.flatten()
sup_flat = I.sup.flatten()

print("\nManual containment check:")
for i in range(min(10, len(c_flat))):
    within = inf_flat[i] <= c_flat[i] <= sup_flat[i]
    print(f"Dim {i}: {inf_flat[i]} <= {c_flat[i]} <= {sup_flat[i]} = {within}")

try:
    res, cert, scaling = I.contains_(test_points)
    print(f"\nResult: {res}")
    print(f"Cert: {cert}")
    print(f"Scaling: {scaling}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc() 