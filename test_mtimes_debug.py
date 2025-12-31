"""
Test mtimes for the specific case: row vector @ interval vector
a = [0, 4] (1, 2) row vector
res = Interval([1, 1], [3, 3]) (2,) interval vector
Expected: a @ res = 0*[1,3] + 4*[1,3] = [4, 12] (scalar interval)
"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.interval.mtimes import mtimes

# Test case from contractInterval
a = np.array([0., 4.])  # (2,) 1D array
print(f"a: {a}, shape={a.shape}, ndim={a.ndim}")

res = Interval([1., 1.], [3., 3.])  # (2,) interval
print(f"res: inf={res.inf}, sup={res.sup}, dim={res.dim()}, inf.ndim={res.inf.ndim}")

# Reshape a to row vector (1, 2)
a_row = a.reshape(1, -1)
print(f"a_row: {a_row}, shape={a_row.shape}, ndim={a_row.ndim}")

# Test: a_row @ res
print("\n=== Testing a_row @ res ===")
result = a_row @ res
print(f"result type: {type(result)}")
if isinstance(result, Interval):
    print(f"result: inf={result.inf}, sup={result.sup}, dim={result.dim()}")
    print(f"result.inf.shape={result.inf.shape}, result.inf.ndim={result.inf.ndim}")
    print(f"result.sup.shape={result.sup.shape}, result.sup.ndim={result.sup.ndim}")

# Expected: scalar interval [4, 12]
print(f"\nExpected: scalar interval [4, 12]")
print(f"Got: {result}")

# Test direct mtimes call
print("\n=== Testing mtimes(a_row, res) ===")
result2 = mtimes(a_row, res)
print(f"result2 type: {type(result2)}")
if isinstance(result2, Interval):
    print(f"result2: inf={result2.inf}, sup={result2.sup}, dim={result2.dim()}")
    print(f"result2.inf.shape={result2.inf.shape}, result2.inf.ndim={result2.inf.ndim}")

