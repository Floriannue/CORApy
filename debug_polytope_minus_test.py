"""Debug polytope minus test to verify correct formula"""
import numpy as np
from cora_python.contSet.polytope import Polytope

# Test case from failing test
A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b = np.ones((4, 1))
p = Polytope(A, b)
v = np.array([2, -3])

print("Original polytope P:")
print(f"  A:\n{A}")
print(f"  b:\n{b}")
print(f"  Constraints: Ax <= b")
print(f"    x <= 1")
print(f"    -x <= 1  => x >= -1")
print(f"    y <= 1")
print(f"    -y <= 1  => y >= -1")
print(f"  So P = [-1, 1] x [-1, 1]")

print(f"\nSubtracting v = {v}")
print(f"  P - v = {{x - v : x in P}}")
print(f"  = {{x' : x' + v in P}}")
print(f"  = {{x' : A(x' + v) <= b}}")
print(f"  = {{x' : Ax' <= b - Av}}")

p_minus_v = p - v
print(f"\nResult:")
print(f"  p_minus_v.b:\n{p_minus_v.b}")

# Calculate expected b
b_expected_minus = b - A @ v.reshape(-1, 1)
b_expected_plus = b + A @ v.reshape(-1, 1)

print(f"\nExpected b (b - A*v):\n{b_expected_minus}")
print(f"\nTest expects (b + A*v):\n{b_expected_plus}")

print(f"\nActual matches b - A*v: {np.allclose(p_minus_v.b, b_expected_minus)}")
print(f"Actual matches b + A*v: {np.allclose(p_minus_v.b, b_expected_plus)}")

# Verify geometrically
print("\nGeometric verification:")
print(f"  Original P: x in [-1, 1], y in [-1, 1]")
print(f"  P - v where v = [2, -3]:")
print(f"    x' = x - 2, so x' in [-1-2, 1-2] = [-3, -1]")
print(f"    y' = y - (-3) = y + 3, so y' in [-1+3, 1+3] = [2, 4]")
print(f"  So P - v: x' in [-3, -1], y' in [2, 4]")
print(f"\n  Constraints for P - v:")
print(f"    x' <= -1  => x' + 1 <= 0  => x' <= -1")
print(f"    -x' <= 3  => -x' - 3 <= 0  => x' >= -3")
print(f"    y' <= 4  => y' - 4 <= 0  => y' <= 4")
print(f"    -y' <= -2  => -y' + 2 <= 0  => y' >= 2")
print(f"\n  So b should be: [-1, 3, 4, -2]")
print(f"  Actual b: {p_minus_v.b.flatten()}")

