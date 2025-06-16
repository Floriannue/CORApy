import sys
sys.path.insert(0, 'cora_python')
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.interval.contains_ import contains_

I = Interval([-2, -1], [3, 4])

# Call contains_ directly
result = contains_(I, [3, 4], method='exact', tol=1e-12, maxEval=200, certToggle=True, scalingToggle=True)
print(f"contains_() result: {result}")
print(f"Result types: {[type(r) for r in result]}")

# Unpack like the contains function does
res, cert, scaling = result
print(f"res: {res} (type: {type(res)})")
print(f"cert: {cert} (type: {type(cert)})")
print(f"scaling: {scaling} (type: {type(scaling)})")

# Check the conditions in my contains function
import numpy as np

print(f"\nChecking contains() logic:")
print(f"isinstance(res, np.ndarray): {isinstance(res, np.ndarray)}")
if isinstance(res, np.ndarray):
    print(f"res.size: {res.size}")
    print(f"res.ndim: {res.ndim}")
    print(f"len(res): {len(res)}")
    
    # First condition: res.size == 1
    if res.size == 1:
        print(f"First condition met: return res.item() = {res.item()}")
    else:
        # Second condition
        if res.ndim == 1 and len(res) > 0:
            print(f"Second condition check:")
            print(f"res[0]: {res[0]}")
            print(f"np.all(res == res[0]): {np.all(res == res[0])}")
            if np.all(res == res[0]):
                result_val = res[0].item() if hasattr(res[0], 'item') else res[0]
                print(f"Second condition met: return {result_val}")
            else:
                print(f"Second condition not met: return res = {res}")
else:
    print(f"res is not numpy array: return res = {res}") 