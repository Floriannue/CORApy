"""
Debug script to compare FGSM attack and verification logic between Python and MATLAB
"""
import os
import sys
import numpy as np

# Add paths - debug script is in root, so cora_root is current directory
cora_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cora_root, 'cora_python'))

# Import aux_readNetworkAndOptions from test file
test_file_path = os.path.join(cora_root, 'cora_python', 'tests', 'nn', 'neuralNetwork', 'test_neuralNetwork_verify.py')
import importlib.util
spec = importlib.util.spec_from_file_location("test_neuralNetwork_verify", test_file_path)
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)
aux_readNetworkAndOptions = test_module.aux_readNetworkAndOptions

def debug_fgsm_verification():
    """Debug the first iteration of FGSM falsification"""
    # Load the same test case as the failing test
    # The test uses ACASXU_run2a_1_2_batch_2000.onnx (not 1_1)
    model1Path = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx')
    prop1Filename = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'prop_1.vnnlib')
    
    if not os.path.isfile(model1Path) or not os.path.isfile(prop1Filename):
        print(f"Files not found:")
        print(f"  model1Path: {model1Path} (exists: {os.path.isfile(model1Path)})")
        print(f"  prop1Filename: {prop1Filename} (exists: {os.path.isfile(prop1Filename)})")
        return
    
    # Load network and options
    nn, options, x, r, A, b, safeSet = aux_readNetworkAndOptions(model1Path, prop1Filename)
    
    print(f"=== Test Case Setup ===")
    print(f"safeSet: {safeSet}")
    print(f"A shape: {A.shape}, A:\n{A}")
    print(f"b shape: {b.shape}, b: {b.flatten()}")
    print(f"x shape: {x.shape}, x: {x.flatten()}")
    print(f"r shape: {r.shape}, r: {r.flatten()}")
    
    # Set options for FGSM
    options['nn']['falsification_method'] = 'fgsm'
    options['nn']['refinement_method'] = 'naive'
    
    # Run first iteration manually to see what happens
    from cora_python.nn.neuralNetwork.verify import verify
    
    # Run with verbose to see debug output
    result, x_, y_ = verify(nn, x, r, A, b, safeSet, options, timeout=10, verbose=True)
    
    print(f"\n=== Verification Result ===")
    print(f"Result: {result}")
    if x_ is not None:
        print(f"x_ (counterexample): {x_.flatten()}")
        print(f"y_ (output): {y_.flatten()}")
        
        # Check if counterexample is valid
        yi_check = nn.evaluate(x_)
        ld_check = A @ yi_check
        print(f"\n=== Counterexample Validation ===")
        print(f"yi_check: {yi_check.flatten()}")
        print(f"A*yi_check: {ld_check.flatten()}")
        print(f"b: {b.flatten()}")
        print(f"A*yi_check - b: {(ld_check - b).flatten()}")
        
        if safeSet:
            violates = np.any(ld_check > b)
            print(f"Violates (safeSet): {violates} (any(A*y > b))")
        else:
            violates = np.all(ld_check <= b)
            print(f"Violates (unsafeSet): {violates} (all(A*y <= b))")
        
        # Check bounds
        in_bounds = np.all(x_ >= x - r) and np.all(x_ <= x + r)
        print(f"\n=== Bounds Check ===")
        print(f"x_ within bounds [x-r, x+r]: {in_bounds}")
        print(f"x - r: {(x - r).flatten()}")
        print(f"x + r: {(x + r).flatten()}")
        print(f"x_: {x_.flatten()}")
    else:
        print("No counterexample found")

if __name__ == '__main__':
    debug_fgsm_verification()

