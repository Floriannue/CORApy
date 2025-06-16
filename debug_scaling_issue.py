import sys
sys.path.insert(0, 'cora_python')
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.interval.contains_ import contains_

I = Interval([-2, -1], [3, 4])

print("Testing [3, 4] with different scaling settings:")

# Test without scaling
print("\n1. Without scaling:")
result1 = contains_(I, [3, 4], method='exact', tol=1e-12, maxEval=200, certToggle=False, scalingToggle=False)
print(f"   Result: {result1}")

# Test with cert but no scaling
print("\n2. With cert, no scaling:")
result2 = contains_(I, [3, 4], method='exact', tol=1e-12, maxEval=200, certToggle=True, scalingToggle=False)
print(f"   Result: {result2}")

# Test with scaling
print("\n3. With scaling:")
result3 = contains_(I, [3, 4], method='exact', tol=1e-12, maxEval=200, certToggle=True, scalingToggle=True)
print(f"   Result: {result3}")

# Test how the contains function calls it
print("\n4. How contains() calls it:")
from cora_python.contSet.contSet.contains import contains
result4 = contains(I, [3, 4])
print(f"   Result: {result4}")

# Test with explicit parameters to contains
print("\n5. With explicit no scaling:")
result5 = contains(I, [3, 4], return_cert=False, return_scaling=False)
print(f"   Result: {result5}")

print("\nThe issue is likely in the scaling calculation for boundary points.")
print("For a boundary point, the scaling should be 1.0, not > 1.0") 