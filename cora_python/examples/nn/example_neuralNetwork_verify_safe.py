"""
example_neuralNetwork_verify_safe - example for the verification of a 
   neural networks using the function neuralNetwork/verify.

Syntax:
    res = example_neuralNetwork_verify_safe()

Inputs:
    -

Outputs:
    res - string, verification result 
       ['VERIFIED','COUNTEREXAMPLE','UNKNOWN']

References:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Lukas Koller
Written:       18-July-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, Any

# Import CORA Python modules
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions


def example_neuralNetwork_verify_safe() -> str:
    """
    Example for the verification of a neural network using the verify function.
    
    Returns:
        res: Verification result string
    """
    # Set random seed to default (equivalent to MATLAB's rng('default'))
    # MATLAB's rng('default') resets to a specific default state
    # Python's np.random.seed() without arguments uses system time
    # For reproducibility, we'll use a fixed seed that represents "default"
    np.random.seed(42)  # This provides consistent behavior similar to MATLAB's default
    
    verbose = True
    # Specify model and specification path.
    # Get the directory where this script is located
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    modelPath = os.path.join(script_dir, 'models', 'ACASXU_run2a_1_2_batch_2000.onnx')
    specPath = os.path.join(script_dir, 'models', 'prop_1.vnnlib')
    timeout = 2
    # Load model and specification.
    nn, x, r, A, b, safeSet, options = aux_readModelAndSpecs(modelPath, specPath)
    # Do verification.
    timerVal = time.time()
    res, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    # Print result.
    if verbose:
        # Print result.
        print(f'{modelPath} -- {specPath}: {res}')
        elapsed_time = time.time() - timerVal
        print(f'--- Verification time: {elapsed_time:.4f} / {timeout:.4f} [s]')
    # Write results.
    print('Result:')
    aux_writeResults(res, x_, y_)
    
    return res


# Auxiliary functions -----------------------------------------------------

def aux_readModelAndSpecs(modelPath: str, specPath: str) -> Tuple[NeuralNetwork, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    """
    Read model and specifications from files.
    
    Args:
        modelPath: Path to the ONNX model file
        specPath: Path to the VNNLIB specification file
        
    Returns:
        Tuple of (nn, x, r, A, b, safeSet, options)
    """
    # Load the model.
    nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BSSC')
    # Load specification.
    X0, specs = vnnlib2cora(specPath)
    # Compute center and radius of the input set.
    # Note: MATLAB uses 1-based indexing, Python uses 0-based
    # MATLAB: X0{1}.sup, Python: X0[0].sup
    # X0[0] is already an Interval object, so we can access .sup and .inf directly
    print(f"DEBUG: X0[0] type: {type(X0[0])}")
    print(f"DEBUG: X0[0].sup type: {type(X0[0].sup)}, shape: {X0[0].sup.shape if hasattr(X0[0].sup, 'shape') else 'no shape'}")
    print(f"DEBUG: X0[0].inf type: {type(X0[0].inf)}, shape: {X0[0].inf.shape if hasattr(X0[0].inf, 'shape') else 'no shape'}")
    
    x = 1/2 * (X0[0].sup + X0[0].inf)
    r = 1/2 * (X0[0].sup - X0[0].inf)
    
    # Ensure x and r are 2D column vectors (n x 1) as expected by verify function
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    
    print(f"DEBUG: x type: {type(x)}, shape: {x.shape if hasattr(x, 'shape') else 'no shape'}")
    print(f"DEBUG: r type: {type(r)}, shape: {r.shape if hasattr(r, 'shape') else 'no shape'}")

    # Extract specification.
    # MATLAB: isa(specs.set,'halfspace')
    # Python: Check if it's a halfspace by looking for 'c' and 'd' attributes
    # This is the most robust way to detect halfspace objects in Python
    if hasattr(specs.set, 'c') and hasattr(specs.set, 'd'):
        # halfspace case
        A = specs.set.c.T
        b = -specs.set.d
    else:
        # polytope case
        A = specs.set.A
        b = -specs.set.b
    
    safeSet = (specs.type == 'safeSet')

    # Create evaluation options.
    options = {}
    options['nn'] = {
        'use_approx_error': True,
        'poly_method': 'bounds',  # 'bounds','singh'
        'num_generators': 100,  # Add this to avoid the None error
        'train': {
            'backprop': False,
            'mini_batch_size': 512
        }
    }
    # Set default training parameters
    options = validateNNoptions(options, True)
    options['nn']['interval_center'] = False
    
    return nn, x, r, A, b, safeSet, options


def aux_writeResults(res: str, x_: Optional[np.ndarray], y_: Optional[np.ndarray]) -> None:
    """
    Write verification results in the expected format.
    
    Args:
        res: Verification result string
        x_: Counterexample input (if found)
        y_: Counterexample output (if found)
    """
    # Write results.
    if res == 'VERIFIED':
        # Write content.
        print('unsat')
    elif res == 'COUNTEREXAMPLE':
        # Write content.
        print('sat')
        print('(')
        # Write input values.
        if x_ is not None:
            for j in range(x_.shape[0]):
                print(f'(X_{j} {x_[j]:f})')
        # Write output values.
        if y_ is not None:
            for j in range(y_.shape[0]):
                print(f'(Y_{j} {y_[j]:f})')
        print(')', end='')  # Print closing parenthesis without newline to match MATLAB
    else:
        print('unknown')


if __name__ == "__main__":
    # Run the example
    result = example_neuralNetwork_verify_safe()
    print(f"Verification result: {result}")
