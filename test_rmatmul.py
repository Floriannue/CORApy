"""
Test __rmatmul__ behavior
"""
import numpy as np
from cora_python.contSet.interval.interval import Interval

a_row = np.array([[0., 4.]])  # (1, 2) numeric
res = Interval([1., 1.], [3., 3.])  # (2,) interval

print(f"a_row: {a_row}, type={type(a_row)}")
print(f"res: {res}, type={type(res)}")

# Test: a_row @ res
print("\n=== a_row @ res ===")
result1 = a_row @ res
print(f"result1: {result1}, type={type(result1)}")
if isinstance(result1, Interval):
    print(f"  inf={result1.inf}, sup={result1.sup}, dim={result1.dim()}")

# Test: res.__rmatmul__(a_row)
print("\n=== res.__rmatmul__(a_row) ===")
result2 = res.__rmatmul__(a_row)
print(f"result2: {result2}, type={type(result2)}")
if isinstance(result2, Interval):
    print(f"  inf={result2.inf}, sup={result2.sup}, dim={result2.dim()}")

# Test: mtimes(a_row, res)
print("\n=== mtimes(a_row, res) ===")
from cora_python.contSet.interval.mtimes import mtimes
result3 = mtimes(a_row, res)
print(f"result3: {result3}, type={type(result3)}")
if isinstance(result3, Interval):
    print(f"  inf={result3.inf}, sup={result3.sup}, dim={result3.dim()}")

