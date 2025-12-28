"""Debug degenerate set buffering"""
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval

# Test case
c1 = np.array([[5.000], [0.000]])
G1 = np.array([[0.000], [1.000]])
Z1 = Zonotope(c1, G1)

c2 = np.array([[5.650], [0.000]])
G2 = np.array([[0.000, 0.050, 0.000, 0.000, 0.000],
               [0.937, 0.000, -0.005, -0.000, 0.000]])
Z2 = Zonotope(c2, G2)

tol = 1e-12

print("Z1 (degenerate):")
print(f"  Z1.c: {Z1.c.flatten()}")
print(f"  Z1.G:\n{Z1.G}")
is_full_dim_result = Z1.isFullDim(tol)
if isinstance(is_full_dim_result, tuple):
    is_full_dim = is_full_dim_result[0]
else:
    is_full_dim = is_full_dim_result
print(f"  Z1.isFullDim: {is_full_dim}")

if not is_full_dim:
    d = Z1.G.shape[0]
    I = tol * Interval(-np.ones((d, 1)), np.ones((d, 1)))
    print(f"\nBuffering Z1 with I:")
    print(f"  I.inf: {I.inf.flatten()}")
    print(f"  I.sup: {I.sup.flatten()}")
    Z1_buffered = Z1 + I
    print(f"  Z1_buffered.c: {Z1_buffered.c.flatten()}")
    print(f"  Z1_buffered.G:\n{Z1_buffered.G}")
    
    # Check Z1_buffered interval
    Z1_int = Z1_buffered.interval()
    print(f"\nZ1_buffered interval:")
    print(f"  Z1.inf: {Z1_int.inf.flatten()}")
    print(f"  Z1.sup: {Z1_int.sup.flatten()}")

print("\nZ2:")
print(f"  Z2.c: {Z2.c.flatten()}")
Z2_int = Z2.interval()
print(f"  Z2.inf: {Z2_int.inf.flatten()}")
print(f"  Z2.sup: {Z2_int.sup.flatten()}")

# Check containment manually
print("\nManual containment check:")
if not is_full_dim:
    Z1_int_buf = Z1_buffered.interval()
    x_contained = (Z1_int_buf.inf[0] <= Z2_int.inf[0] + tol) and (Z1_int_buf.sup[0] >= Z2_int.sup[0] - tol)
    y_contained = (Z1_int_buf.inf[1] <= Z2_int.inf[1] + tol) and (Z1_int_buf.sup[1] >= Z2_int.sup[1] - tol)
    print(f"  x-contained: {x_contained}")
    print(f"  y-contained: {y_contained}")
    print(f"  Should be contained: {x_contained and y_contained}")

