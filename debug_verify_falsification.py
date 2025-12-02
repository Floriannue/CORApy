"""
Debug script to compare Python and MATLAB verify falsification logic
"""
import numpy as np
import os
from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.tests.nn.neuralNetwork.test_neuralNetwork_verify import aux_readNetworkAndOptions

# Load the same test case
cora_root = CORAROOT()
model1Path = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx')
prop1Filename = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'prop_1.vnnlib')

if os.path.isfile(model1Path) and os.path.isfile(prop1Filename):
    nn, options, x, r, A, b, safeSet = aux_readNetworkAndOptions(model1Path, prop1Filename)
    
    # Set options to match MATLAB test
    options['nn']['falsification_method'] = 'fgsm'
    options['nn']['refinement_method'] = 'naive'
    
    print(f"Input x shape: {x.shape}, r shape: {r.shape}")
    print(f"A shape: {A.shape}, b shape: {b.shape}, safeSet: {safeSet}")
    print(f"b value: {b}")
    
    # Run verification with verbose to see what's happening
    timeout = 10
    verbose = True
    res, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    
    print(f"\nResult: {res}")
    print(f"x_ is None: {x_ is None}, y_ is None: {y_ is None}")
    if x_ is not None:
        print(f"x_ shape: {x_.shape}, x_ value: {x_.flatten()}")
    if y_ is not None:
        print(f"y_ shape: {y_.shape}, y_ value: {y_.flatten()}")
        
    # Check if counterexample is valid
    if x_ is not None and y_ is not None:
        # Check if x_ is within input bounds
        x_lower = x - r
        x_upper = x + r
        print(f"\nInput bounds check:")
        print(f"x (center): {x.flatten()}")
        print(f"r (radius): {r.flatten()}")
        print(f"x_lower: {x_lower.flatten()}")
        print(f"x_upper: {x_upper.flatten()}")
        print(f"x_ (counterexample): {x_.flatten()}")
        in_bounds = np.all((x_ >= x_lower) & (x_ <= x_upper))
        print(f"x_ within bounds: {in_bounds}")
        if not in_bounds:
            print(f"OUT OF BOUNDS! Violations:")
            print(f"  Below lower: {(x_ < x_lower).flatten()}")
            print(f"  Above upper: {(x_ > x_upper).flatten()}")
        
        yi = nn.evaluate(x_)
        ld_yi = A @ yi
        print(f"\nCounterexample check:")
        print(f"yi from evaluate: {yi.flatten()}")
        print(f"y_ from verify: {y_.flatten()}")
        print(f"Difference: {np.abs(y_ - yi).flatten()}")
        print(f"A*yi: {ld_yi.flatten()}")
        print(f"b: {b.flatten()}")
        print(f"A*yi - b: {(ld_yi - b).flatten()}")
        if safeSet:
            violates = np.any(ld_yi > b)
            print(f"Violates (safeSet): {violates} (any(A*y > b))")
        else:
            violates = np.all(ld_yi <= b)
            print(f"Violates (unsafeSet): {violates} (all(A*y <= b))")
else:
    print("Files not found, skipping debug")

