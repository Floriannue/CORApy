import sys
sys.path.insert(0, 'cora_python')
import numpy as np
from cora_python.contSet.interval.interval import Interval

# Test the specific boundary case
I = Interval([-2, -1], [3, 4])

# Test with exact boundaries
points_to_test = [
    ([3, 4], "upper boundary - should be True"),
    ([-2, -1], "lower boundary - should be True"), 
    ([0, 0], "inside - should be True"),
    ([3.1, 4], "outside upper - should be False"),
    ([3, 4.1], "outside upper - should be False"),
]

print("Testing boundary cases:")
for point, description in points_to_test:
    result = I.contains(point)
    print(f"I.contains({point}) = {result} ({description})")
    
# Test direct call to contains_ for debugging
from cora_python.contSet.interval.contains_ import contains_

print("\nDirect contains_ call for [3, 4]:")
result_direct = contains_(I, [3, 4], method='exact', tol=1e-12, maxEval=200, certToggle=False, scalingToggle=False)
print(f"contains_() result: {result_direct}")

# Check with a larger tolerance
print(f"\nWith larger tolerance (1e-6):")
result_larger_tol = contains_(I, [3, 4], method='exact', tol=1e-6, maxEval=200, certToggle=False, scalingToggle=False)
print(f"contains_() result: {result_larger_tol}")

# Test very simple manual check
print(f"\nManual check:")
inf = I.inf
sup = I.sup  
point = np.array([3.0, 4.0])
print(f"inf <= point: {inf <= point}")
print(f"point <= sup: {point <= sup}")
print(f"all(inf <= point <= sup): {np.all(inf <= point) and np.all(point <= sup)}") 