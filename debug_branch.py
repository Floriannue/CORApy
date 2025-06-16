#!/usr/bin/env python3

import sys
sys.path.insert(0, 'cora_python')

import numpy as np
from cora_python.contSet.interval.interval import Interval

# Create interval
I = Interval([-2, -1], [3, 4])
point = [0, 0]
S = np.asarray(point, dtype=float)

print(f"I.inf shape: {I.inf.shape}")
print(f"I.sup shape: {I.sup.shape}")
print(f"S: {S}")
print(f"S.shape: {S.shape}")
print(f"S.ndim: {S.ndim}")
print(f"len(I.inf.shape): {len(I.inf.shape)}")

# Check conditions
print(f"\nBranch conditions:")
print(f"isinstance(S, (int, float, list, tuple, np.ndarray)): {isinstance(S, (int, float, list, tuple, np.ndarray))}")

if isinstance(S, (int, float, list, tuple, np.ndarray)):
    print("==> Going into point containment branch")
    
    cond1 = S.ndim == len(I.inf.shape) + 1
    print(f"S.ndim == len(I.inf.shape) + 1: {S.ndim} == {len(I.inf.shape) + 1} = {cond1}")
    
    if cond1:
        print("==> Would go into multi-dimensional case")
    else:
        cond2a = S.ndim == len(I.inf.shape)
        cond2b = S.shape != I.inf.shape
        cond2 = cond2a and cond2b
        print(f"S.ndim == len(I.inf.shape): {S.ndim} == {len(I.inf.shape)} = {cond2a}")
        print(f"S.shape != I.inf.shape: {S.shape} != {I.inf.shape} = {cond2b}")
        print(f"Combined condition 2: {cond2}")
        
        if cond2:
            print("==> Would go into regular point array case")
        else:
            print("==> Would go into single point case (else branch)")
            
    # Test both cases
    from cora_python.contSet.interval.contains_ import contains_
    
    print(f"\n--- Test 1: No scaling/cert ---")
    result1 = contains_(I, S, method='exact', tol=1e-12, maxEval=200, certToggle=False, scalingToggle=False)
    print(f"Result: {result1}")
    print(f"Types: {[type(r) for r in result1]}")
    
    print(f"\n--- Test 2: With scaling/cert ---")
    result2 = contains_(I, S, method='exact', tol=1e-12, maxEval=200, certToggle=True, scalingToggle=True)
    print(f"Result: {result2}")
    print(f"Types: {[type(r) for r in result2]}") 