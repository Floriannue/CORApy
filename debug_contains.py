#!/usr/bin/env python3

import sys
sys.path.insert(0, 'cora_python')

import numpy as np
from cora_python.contSet.interval.interval import Interval

# Create interval
I = Interval([-2, -1], [3, 4])

print(f"Interval I: inf={I.inf}, sup={I.sup}")
print(f"I.inf shape: {I.inf.shape}, I.sup shape: {I.sup.shape}")

# Test point
point = [0, 0]
print(f"Test point: {point}")

print("\n=== Step by step debugging ===")

# Step 1: Call contains_ directly
from cora_python.contSet.interval.contains_ import contains_
result = contains_(I, point, method='exact', tol=1e-12, maxEval=200, certToggle=True, scalingToggle=True)
print(f"1. contains_ result: {result}")
print(f"   Result types: {[type(r) for r in result]}")

# Step 2: Unpack the result like contains function does
res, cert, scaling = result
print(f"2. Unpacked: res={res} (type: {type(res)}), cert={cert} (type: {type(cert)}), scaling={scaling} (type: {type(scaling)})")

# Step 3: Check what contains function does step by step
print(f"3. In contains function, return_cert=False, return_scaling=False, so should return just res")
print(f"   Just res: {res} (type: {type(res)})")

# Step 4: Call contains directly
from cora_python.contSet.contSet.contains import contains
result2 = contains(I, point)
print(f"4. contains function result: {result2} (type: {type(result2)})")

# Step 5: Call method on interval
result3 = I.contains(point)
print(f"5. I.contains method result: {result3} (type: {type(result3)})")

# Step 6: Try manual call to see difference 
result4 = contains(I, point, return_cert=False, return_scaling=False)
print(f"6. contains with explicit flags: {result4} (type: {type(result4)})") 