#!/usr/bin/env python3
"""
Debug script to test if fpolyder modifies input coefficients
"""

import sys
sys.path.append('cora_python')
from nn.nnHelper.fpolyder import fpolyder
import numpy as np

# Test coefficients
coeffs = np.array([0.45362903, 0.50000000, 0.09698276])

print("=== fpolyder Modification Test ===")
print(f"Initial coeffs: {coeffs}")
print(f"Initial coeffs id: {id(coeffs)}")

# Test fpolyder
dp = fpolyder(coeffs)
print(f"After fpolyder - coeffs: {coeffs}")
print(f"After fpolyder - coeffs id: {id(coeffs)}")
print(f"fpolyder result: {dp}")
print(f"fpolyder result id: {id(dp)}")

# Test if the issue is in getDerInterval
print("\n=== getDerInterval Test ===")
from nn.nnHelper.getDerInterval import getDerInterval

coeffs2 = np.array([0.45362903, 0.50000000, 0.09698276])
print(f"Before getDerInterval - coeffs: {coeffs2}")
print(f"Before getDerInterval - coeffs id: {id(coeffs2)}")

der2l, der2u = getDerInterval(coeffs2, -1, 1)
print(f"After getDerInterval - coeffs: {coeffs2}")
print(f"After getDerInterval - coeffs id: {id(coeffs2)}")
print(f"getDerInterval result: der2l={der2l:.8f}, der2u={der2u:.8f}")

# Let's also test the specific case from our debug
print("\n=== Specific Case Test ===")
coeffs3 = np.array([0.45362903, 0.50000000, 0.09698276])
print(f"Before any operations - coeffs: {coeffs3}")
print(f"Before any operations - coeffs id: {id(coeffs3)}")

# Simulate what happens in getDerInterval
p = coeffs3  # This creates a reference!
print(f"After p = coeffs3 - coeffs3: {coeffs3}")
print(f"After p = coeffs3 - p: {p}")
print(f"After p = coeffs3 - p id: {id(p)}")

dp = fpolyder(p)
print(f"After fpolyder(p) - coeffs3: {coeffs3}")
print(f"After fpolyder(p) - p: {p}")
print(f"After fpolyder(p) - dp: {dp}")

dp2 = fpolyder(dp)
print(f"After fpolyder(dp) - coeffs3: {coeffs3}")
print(f"After fpolyder(dp) - p: {p}")
print(f"After fpolyder(dp) - dp2: {dp2}")
