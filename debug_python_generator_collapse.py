"""
Debug script to compare r, ri_, and m values with MATLAB
This uses the REAL verify() function with debug logging enabled
"""

import numpy as np
import sys
import os

# Add cora_python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.neuralNetwork.verify import verify
from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions

def debug_python_generator_collapse():
    """Debug script matching MATLAB debug_matlab_generator_collapse.m
    Uses the REAL verify() function with debug logging enabled via options
    """
    
    print("=== Python Generator Collapse Debug Script ===\n")
    
    # Paths (matching MATLAB script)
    modelPath = os.path.join('cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx')
    specPath = os.path.join('cora_matlab', 'models', 'Cora', 'nn', 'prop_1.vnnlib')
    
    print("Loading network and specification...\n")
    
    # Read network and options (matching test)
    options = {}
    options['nn'] = {
        'use_approx_error': False,
        'poly_method': 'bounds',
        'train': {
            'backprop': False,
            'mini_batch_size': 32,
            'num_init_gens': 5,
            'num_approx_err': 0
        }
    }
    options = validateNNoptions(options, True)
    options['nn']['interval_center'] = False
    options['nn']['refinement_method'] = 'naive'
    options['nn']['falsification_method'] = 'fgsm'
    
    # Enable debug logging (this will be passed through to helper functions)
    # The debug logging in verify_helpers.py checks for _debug_iteration in options
    # We'll enable it for first few iterations
    options['_enable_debug'] = True
    options['_debug_max_iterations'] = 1  # Only debug first iteration
    
    # Read network
    nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BSSC')
    print(f"Network loaded: {len(nn.layers)} layers")
    
    # Read specification (using same approach as test)
    X0, specs = vnnlib2cora(specPath)
    x = 0.5 * (X0[0].sup + X0[0].inf)
    r = 0.5 * (X0[0].sup - X0[0].inf)
    
    # Extract specification (matching test code exactly)
    from cora_python.contSet.polytope.representsa_ import representsa_
    isHalfspace = representsa_(specs.set, 'halfspace')
    
    if isHalfspace:
        # Halfspace case: c' * x <= d
        # MATLAB: A = specs.set.c';
        # MATLAB: b = specs.set.d;
        # In Python, a halfspace polytope has A (1, n) and b (1, 1)
        A = specs.set.A  # (1, n) - this is c' in MATLAB
        b = specs.set.b.flatten()  # (1,) - this is d in MATLAB
        if b.ndim == 0:
            b = np.array([b])
    else:
        # Polytope case: A * x <= b
        # MATLAB: A = specs.set.A;
        A = specs.set.A
        # MATLAB: b = specs.set.b;
        b = specs.set.b
        if b.ndim == 1:
            b = b.reshape(-1, 1)
    
    # MATLAB: safeSet = strcmp(specs.type,'safeSet');
    safeSet = (specs.type == 'safeSet')
    
    print(f"Input dimension: {x.shape[0]}")
    print(f"Specification: A shape {A.shape}, b shape {b.shape}, safeSet={safeSet}\n")
    
    print("==== Starting Verification (using REAL verify() function) ===\n")
    print("Note: Debug logging will appear from verify_helpers.py and nnActivationLayer.py\n")
    print("=" * 60 + "\n")
    
    # Use the REAL verify() function with verbose=True to see debug output
    # The debug logging we added will automatically print when _debug_iteration is set
    timeout = 10.0
    verbose = True  # Enable verbose output
    
    # Call the real verify function - this will use all the real implementations
    verifRes, x_, y_ = verify(nn, x, r, A, b, safeSet, options, timeout, verbose)
    
    print("\n" + "=" * 60)
    print(f"=== Verification Result: {verifRes} ===")
    print("=== Debug Complete ===\n")

if __name__ == '__main__':
    debug_python_generator_collapse()

