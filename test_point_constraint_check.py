"""Test if point interval satisfies constraint"""
import numpy as np
from cora_python.contSet.interval.interval import Interval

def f(x):
    return x[0]**2 + x[1]**2 - 4

# Test the point interval
dom_point = Interval([1.0981, 1.0981], [1.0981, 1.0981])
print(f"Point interval: {dom_point}")

# Evaluate constraint
temp = f(dom_point)
print(f"f(dom_point): {temp}")

# Check if it contains 0
p = np.zeros(len(temp) if hasattr(temp, '__len__') else 1)
print(f"p: {p}")

contains_result = temp.contains_(p)
print(f"temp.contains_(p): {contains_result}")

# Check the actual value
point_val = f([1.0981, 1.0981])
print(f"\nf([1.0981, 1.0981]) = {point_val}")
print(f"abs(point_val) < 1e-6: {abs(point_val) < 1e-6}")
print(f"abs(point_val) < 1e-3: {abs(point_val) < 1e-3}")

# What's the actual solution?
import math
sqrt2 = math.sqrt(2)
print(f"\nActual solution: [sqrt(2), sqrt(2)] = [{sqrt2}, {sqrt2}]")
print(f"f([sqrt(2), sqrt(2)]) = {sqrt2**2 + sqrt2**2 - 4}")

