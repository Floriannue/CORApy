"""
Test Python mtimes to find the bug
"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.interval.mtimes import _mtimes_nonsparse

# Test case: numeric (1, 2) @ interval (2,)
factor1 = Interval(np.array([[0., 4.]]), np.array([[0., 4.]]))  # (1, 2) interval (from numeric)
factor2 = Interval(np.array([1., 1.]), np.array([3., 3.]))  # (2,) interval

print(f"factor1.inf: {factor1.inf}, shape={factor1.inf.shape}, ndim={factor1.inf.ndim}")
print(f"factor2.inf: {factor2.inf}, shape={factor2.inf.shape}, ndim={factor2.inf.ndim}")

# Python code path
# Line 258-267: Get m and k1
if factor1.inf.ndim == 1:
    m = 1
    f1_inf = factor1.inf.reshape(1, -1)
    f1_sup = factor1.sup.reshape(1, -1)
    k1 = f1_inf.shape[1]
else:
    m = factor1.inf.shape[0]
    f1_inf = factor1.inf
    f1_sup = factor1.sup
    k1 = f1_inf.shape[1] if f1_inf.ndim >= 2 else 1

print(f"\nm={m}, k1={k1}")
print(f"f1_inf: {f1_inf}, shape={f1_inf.shape}")

# Line 346-356: factor2 is interval
if factor2.inf.ndim == 1:
    extSize = (1, factor2.inf.shape[0])
    k2 = factor2.inf.shape[0]
    n = 1
else:
    extSize = (1,) + factor2.inf.shape
    k2 = factor2.inf.shape[0]
    n = factor2.inf.shape[1] if factor2.inf.ndim >= 2 else 1

print(f"\nextSize={extSize}, k2={k2}, n={n}")

# Line 364-365: Reshape factor2
f2_inf = factor2.inf.reshape(extSize)
f2_sup = factor2.sup.reshape(extSize)
print(f"f2_inf: {f2_inf}, shape={f2_inf.shape}")
print(f"f2_sup: {f2_sup}, shape={f2_sup.shape}")

# Line 368-373: Create factor1 with trailing dimension
if f1_inf.ndim == 1:
    f1_inf_bc = f1_inf.reshape(1, -1, 1)
    f1_sup_bc = f1_sup.reshape(1, -1, 1)
else:
    f1_inf_bc = f1_inf[:, :, np.newaxis]
    f1_sup_bc = f1_sup[:, :, np.newaxis]

print(f"\nf1_inf_bc: {f1_inf_bc}, shape={f1_inf_bc.shape}")
print(f"f1_sup_bc: {f1_sup_bc}, shape={f1_sup_bc.shape}")

# Line 377-382: Compute products
# f2_inf and f2_sup are (1, 2), need to add dimension for broadcasting
# MATLAB: [m, k, 1] .* [1, k, n]
# f1_inf_bc is (1, 2, 1)
# f2_inf is (1, 2), need to be (1, 1, 2) or (1, 2, 1)?
# Actually, MATLAB reshapes factor2 to extSize = (1, 2), then uses it directly
# So f2_inf should be (1, 2), and for broadcasting it should be (1, 2, 1) or (1, 1, 2)?

# Let's check: MATLAB does factor1 .* factor2 where:
# factor1 is (1, 2) interval
# factor2.inf is (1, 2) after reshape
# So element-wise: (1, 2) .* (1, 2) = (1, 2)
# Then sum along dim 2: (1, 2) -> (1, 1)

# But Python is doing [m, k, 1] .* [1, k, n]
# f1_inf_bc is (1, 2, 1)
# f2_inf needs to be broadcastable: should be (1, 2, 1) or (1, 1, 2)?

# Actually, the issue is that f2_inf is (1, 2) but needs to be reshaped for the 3D broadcasting
# MATLAB doesn't do 3D broadcasting - it does 2D element-wise, then sums
# Python is trying to do 3D broadcasting which is wrong!

# The fix: f2_inf should be used as (1, 2) for element-wise with f1_inf (1, 2), not 3D
print("\n=== The Problem ===")
print("Python is doing 3D broadcasting, but MATLAB does 2D element-wise then sum")
print("f2_inf needs to be (1, 2) not reshaped for 3D")

# Correct approach: f2_inf is (1, 2), f1_inf_bc is (1, 2, 1)
# For proper broadcasting: f2_inf should be (1, 2, 1) to match
f2_inf_bc = f2_inf[:, :, np.newaxis] if f2_inf.ndim == 2 else f2_inf.reshape(extSize + (1,))
f2_sup_bc = f2_sup[:, :, np.newaxis] if f2_sup.ndim == 2 else f2_sup.reshape(extSize + (1,))
print(f"f2_inf_bc: {f2_inf_bc}, shape={f2_inf_bc.shape}")

# Now products: (1, 2, 1) .* (1, 2, 1) = (1, 2, 1)
products = [
    f1_inf_bc * f2_inf_bc,
    f1_inf_bc * f2_sup_bc,
    f1_sup_bc * f2_inf_bc,
    f1_sup_bc * f2_sup_bc
]

all_products = np.stack(products, axis=-1)
print(f"\nall_products shape: {all_products.shape}")

inf_result = np.min(all_products, axis=-1)
sup_result = np.max(all_products, axis=-1)
print(f"After min/max: inf_result shape={inf_result.shape}, sup_result shape={sup_result.shape}")

# Sum along axis 1
inf_result = np.sum(inf_result, axis=1)
sup_result = np.sum(sup_result, axis=1)
print(f"After sum: inf_result={inf_result}, sup_result={sup_result}")

# Reshape
if inf_result.ndim == 1:
    inf_result = inf_result.reshape(1, -1)
    sup_result = sup_result.reshape(1, -1)
else:
    inf_result = inf_result.reshape(m, -1)
    sup_result = sup_result.reshape(m, -1)

print(f"Final: inf={inf_result}, sup={sup_result}")

