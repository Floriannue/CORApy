import numpy as np
from cora_python.nn.nnHelper.minMaxDiffPoly import minMaxDiffPoly

# Test case from the failing scenario
coeffs1 = np.array([-0.1, 0.0])  # p1(x) = -0.1*x + 0
coeffs2 = np.array([0.01, 0.0])  # p2(x) = 0.01*x + 0
l, u = -1, 0

print("Testing fixed minMaxDiffPoly:")
print(f"coeffs1: {coeffs1}")
print(f"coeffs2: {coeffs2}")
print(f"l: {l}, u: {u}")

# Test the fixed implementation
diffl, diffu = minMaxDiffPoly(coeffs1, coeffs2, l, u)
print(f"Result: diffl = {diffl}, diffu = {diffu}")

# Expected MATLAB result: diffl = 0, diffu = 0.11
print(f"Expected MATLAB: diffl = 0, diffu = 0.11")
print(f"Match: {np.isclose(diffl, 0) and np.isclose(diffu, 0.11)}")

# Let's also test the second case (x > 0) - this should be p(x) - 1*x
coeffs3 = np.array([1, 0])  # 1*x + 0
l2, u2 = 0, 1
diffl2, diffu2 = minMaxDiffPoly(coeffs1, coeffs3, l2, u2)
print(f"\nSecond case (l={l2}, u={u2}): p(x) - 1*x")
print(f"coeffs1: {coeffs1}, coeffs3: {coeffs3}")
print(f"Result: diffl = {diffl2}, diffu = {diffu2}")
print(f"Expected MATLAB: diffl = -1.1, diffu = 0")
print(f"Match: {np.isclose(diffl2, -1.1) and np.isclose(diffu2, 0)}")
