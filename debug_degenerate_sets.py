"""Debug degenerate sets test"""
import numpy as np
from cora_python.contSet.zonotope import Zonotope

# Test case from failing test
c1 = np.array([[5.000], [0.000]])
G1 = np.array([[0.000], [1.000]])
Z1 = Zonotope(c1, G1)

c2 = np.array([[5.650], [0.000]])
G2 = np.array([[0.000, 0.050, 0.000, 0.000, 0.000],
               [0.937, 0.000, -0.005, -0.000, 0.000]])
Z2 = Zonotope(c2, G2)

print("Z1 (degenerate):")
print(f"  Z1.c: {Z1.c.flatten()}")
print(f"  Z1.G:\n{Z1.G}")
print(f"  Z1.isFullDim: {Z1.isFullDim(1e-12)}")

print("\nZ2:")
print(f"  Z2.c: {Z2.c.flatten()}")
print(f"  Z2.G:\n{Z2.G}")
print(f"  Z2.isFullDim: {Z2.isFullDim(1e-12)}")

# Check containment
print("\nCalling Z1.contains_(Z2)...")
res, cert, scaling = Z1.contains_(Z2)
print(f"  Result: res={res}, cert={cert}, scaling={scaling}")

# Check Z2 interval bounds
Z2_int = Z2.interval()
print(f"\nZ2 interval bounds:")
print(f"  Z2.inf: {Z2_int.inf.flatten()}")
print(f"  Z2.sup: {Z2_int.sup.flatten()}")

# Z1 is degenerate (1D line), so check if Z2's x-coordinate is contained
print(f"\nZ1 is degenerate along x-axis at x=5.0")
print(f"Z2 x-range: [{Z2_int.inf[0,0]:.6f}, {Z2_int.sup[0,0]:.6f}]")
print(f"Z2 should NOT be contained since x=5.65 is outside Z1's x-range [5.0, 5.0]")

