"""
Test __rmatmul__ behavior for all contSet classes
"""
import numpy as np

# Test Interval
print("=== Testing Interval ===")
from cora_python.contSet.interval.interval import Interval
a_row = np.array([[0., 4.]])
res_interval = Interval([1., 1.], [3., 3.])
result = a_row @ res_interval
print(f"Interval: {result}, type={type(result)}, dim={result.dim() if hasattr(result, 'dim') else 'N/A'}")

# Test Zonotope
print("\n=== Testing Zonotope ===")
from cora_python.contSet.zonotope.zonotope import Zonotope
try:
    c = np.array([[0.], [0.]])
    G = np.array([[1., 0.], [0., 1.]])
    z = Zonotope(c, G)
    a_row_z = np.array([[1., 1.]])
    result_z = a_row_z @ z
    print(f"Zonotope: {result_z}, type={type(result_z)}, dim={result_z.dim() if hasattr(result_z, 'dim') else 'N/A'}")
except Exception as e:
    print(f"Zonotope error: {e}")

# Test Ellipsoid
print("\n=== Testing Ellipsoid ===")
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
try:
    Q = np.eye(2)
    q = np.zeros((2, 1))
    e = Ellipsoid(Q, q)
    a_row_e = np.array([[1., 1.]])
    result_e = a_row_e @ e
    print(f"Ellipsoid: {result_e}, type={type(result_e)}, dim={result_e.dim() if hasattr(result_e, 'dim') else 'N/A'}")
except Exception as e:
    print(f"Ellipsoid error: {e}")

# Test Fullspace
print("\n=== Testing Fullspace ===")
from cora_python.contSet.fullspace.fullspace import Fullspace
try:
    fs = Fullspace(2)
    a_row_fs = np.array([[1., 1.]])
    result_fs = a_row_fs @ fs
    print(f"Fullspace: {result_fs}, type={type(result_fs)}, dim={result_fs.dim() if hasattr(result_fs, 'dim') else 'N/A'}")
except Exception as e:
    print(f"Fullspace error: {e}")

# Test Taylm (if available)
print("\n=== Testing Taylm ===")
try:
    from cora_python.contSet.taylm.taylm import Taylm
    # Taylm requires more complex initialization, skip for now
    print("Taylm: Skipped (requires complex initialization)")
except Exception as e:
    print(f"Taylm error: {e}")

print("\n=== All tests completed ===")

