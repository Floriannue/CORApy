"""
Test to understand MATLAB's logic step by step
MATLAB: a_row (1,2) @ res (2,) interval
"""
import numpy as np
from cora_python.contSet.interval.interval import Interval

# Simulate MATLAB's aux_mtimes_nonsparse for numeric @ interval
# MATLAB: factor1 is numeric (1, 2), factor2 is interval (2,)

# Step 1: MATLAB: [m, ~] = size(factor1);
# factor1 is (1, 2), so m = 1
m = 1

# Step 2: MATLAB: extSize = [1, size(factor2)];
# factor2 is interval with inf/sup shape (2,), so size(factor2) = [2]
# extSize = [1, 2]
extSize = (1, 2)

# Step 3: MATLAB: factor2.inf = reshape(factor2.inf, extSize);
# Reshape (2,) to (1, 2)
factor2_inf = np.array([1., 1.])  # (2,)
factor2_sup = np.array([3., 3.])  # (2,)
f2_inf = factor2_inf.reshape(extSize)  # (1, 2)
f2_sup = factor2_sup.reshape(extSize)  # (1, 2)
print(f"f2_inf: {f2_inf}, shape={f2_inf.shape}")
print(f"f2_sup: {f2_sup}, shape={f2_sup.shape}")

# Step 4: MATLAB: res = factor1 .* factor2;
# factor1 is (1, 2) numeric, factor2 is (1, 2) interval
# Element-wise: (1, 2) .* (1, 2) = (1, 2)
factor1 = np.array([[0., 4.]])  # (1, 2)
print(f"factor1: {factor1}, shape={factor1.shape}")

# Create factor1 with trailing dimension for broadcasting
# MATLAB: [m, k, 1] .* [1, k, n] = [m, k, n]
# factor1 is (1, 2), so reshape to (1, 2, 1)
f1_bc = factor1[:, :, np.newaxis]  # (1, 2, 1)
print(f"f1_bc: shape={f1_bc.shape}")

# f2 is (1, 2), reshape to (1, 1, 2) for broadcasting
f2_inf_bc = f2_inf[np.newaxis, :, :]  # (1, 1, 2) - wait, this is wrong
# Actually, MATLAB reshapes factor2 to extSize = (1, 2), so it's already (1, 2)
# For broadcasting: (1, 2, 1) .* (1, 1, 2) = (1, 2, 2) - this is wrong!

# Let me re-read MATLAB code more carefully
# MATLAB: [m, k, 1] .* [1, k, n]
# factor1 is (m, k) = (1, 2)
# factor2 is reshaped to (1, k, n) = (1, 2, 1) if n=1, or (1, 2, n) if n>1
# But factor2 is (2,), so after reshape to (1, 2), n should be 1
# So: (1, 2, 1) .* (1, 2, 1) = (1, 2, 1) - element-wise

# Actually, let me check: MATLAB line 214: factor2.inf = reshape(factor2.inf, extSize);
# extSize = [1, size(factor2)] = [1, 2]
# So factor2.inf becomes (1, 2)
# Then: factor1 .* factor2 where factor1 is (1, 2) and factor2 is (1, 2) interval
# This gives (1, 2) interval result
# Then: sum(..., 2) sums along dimension 2 (columns), so (1, 2) -> (1, 1)
# Then: reshape(res, m, []) reshapes (1, 1) to (1, 1) or (1,)

# So the correct logic:
f1_inf_bc = factor1[:, :, np.newaxis]  # (1, 2, 1)
f1_sup_bc = factor1[:, :, np.newaxis]  # (1, 2, 1)
f2_inf_bc = f2_inf[:, :, np.newaxis]  # (1, 2, 1) - wait, f2_inf is (1, 2), so this gives (1, 2, 1)
f2_sup_bc = f2_sup[:, :, np.newaxis]  # (1, 2, 1)

# Element-wise: (1, 2, 1) .* (1, 2, 1) = (1, 2, 1)
res_inf = f1_inf_bc * f2_inf_bc  # (1, 2, 1)
res_sup = f1_sup_bc * f2_sup_bc  # (1, 2, 1)
print(f"res_inf: {res_inf}, shape={res_inf.shape}")
print(f"res_sup: {res_sup}, shape={res_sup.shape}")

# MATLAB: sum(..., 2) - sum along dimension 2 (axis 1 in 0-indexed)
inf_result = np.sum(res_inf, axis=1)  # (1, 2, 1) -> (1, 1)
sup_result = np.sum(res_sup, axis=1)  # (1, 2, 1) -> (1, 1)
print(f"inf_result: {inf_result}, shape={inf_result.shape}")
print(f"sup_result: {sup_result}, shape={sup_result.shape}")

# MATLAB: reshape(res, m, []) where m=1
inf_result = inf_result.reshape(1, -1)  # (1, 1)
sup_result = sup_result.reshape(1, -1)  # (1, 1)
print(f"Final inf_result: {inf_result}, shape={inf_result.shape}")
print(f"Final sup_result: {sup_result}, shape={sup_result.shape}")

# Expected: [4, 12] as scalar interval
print(f"\nExpected: [4, 12]")
print(f"Got: [{inf_result[0,0]}, {sup_result[0,0]}]")

