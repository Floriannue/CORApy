#!/usr/bin/env python3

import sys
sys.path.insert(0, 'cora_python')

import numpy as np
from cora_python.contSet.interval.interval import Interval

# Create interval
I = Interval([-2, -1], [3, 4])
point = [0, 0]

# Manually trace through the logic that's in contains_
print("=== Manual trace through contains_ logic ===")

S = np.asarray(point, dtype=float)
method = 'exact'
tol = 1e-12
scalingToggle = False

print(f"S: {S}, S.shape: {S.shape}, S.ndim: {S.ndim}")
print(f"I.inf: {I.inf}, I.inf.shape: {I.inf.shape}")
print(f"isinstance(S, (int, float, list, tuple, np.ndarray)): {isinstance(S, (int, float, list, tuple, np.ndarray))}")

if isinstance(S, (int, float, list, tuple, np.ndarray)):
    print(">>> Point in interval containment")
    
    if scalingToggle:
        print(">>> With scaling (not our case)")
    else:
        print(">>> Without scaling")
        
        if S.ndim == len(I.inf.shape) + 1:
            print(">>> Multi-dimensional case")
        elif S.ndim == len(I.inf.shape) and S.shape != I.inf.shape:
            print(">>> Regular point array case")
        else:
            print(">>> Single point case")
            
            # This is the exact code from the else branch
            from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
            
            lower_check = (I.inf < S + tol) | withinTol(I.inf, S, tol)
            upper_check = (I.sup > S - tol) | withinTol(I.sup, S, tol)
            
            print(f"lower_check: {lower_check}, type: {type(lower_check)}")
            print(f"upper_check: {upper_check}, type: {type(upper_check)}")
            
            # Combine checks
            containment_check = lower_check & upper_check
            print(f"containment_check: {containment_check}, type: {type(containment_check)}")
            
            # For single point, check all dimensions
            res = np.all(containment_check)
            print(f"res = np.all(containment_check): {res}, type: {type(res)}")
            
            cert = True
            scaling = 0.0
            
            print(f"Final: res={res} (type: {type(res)}), cert={cert} (type: {type(cert)}), scaling={scaling} (type: {type(scaling)})")
            
        # Check the return logic
        print(f"\n>>> Return logic:")
        print(f"np.isscalar(res): {np.isscalar(res)}")
        if hasattr(res, 'size'):
            print(f"res.size: {res.size}")
        else:
            print("res has no size attribute")
        
        condition = np.isscalar(res) or (hasattr(res, 'size') and res.size == 1)
        print(f"Return condition: {condition}")
        
        if condition:
            print(">>> Should return scalars")
            res_scalar = res.item() if hasattr(res, 'item') else res
            cert_scalar = cert.item() if hasattr(cert, 'item') else cert
            scaling_scalar = scaling.item() if hasattr(scaling, 'item') else scaling
            print(f"Final scalars: {res_scalar} (type: {type(res_scalar)}), {cert_scalar} (type: {type(cert_scalar)}), {scaling_scalar} (type: {type(scaling_scalar)})")
        else:
            print(">>> Should return arrays (this is wrong for single point!)")

# Now test the actual function
print(f"\n=== Actual function call ===")
from cora_python.contSet.interval.contains_ import contains_
result = contains_(I, point, method='exact', tol=1e-12, maxEval=200, certToggle=False, scalingToggle=False)
print(f"Actual result: {result}")
print(f"Types: {[type(r) for r in result]}") 