"""
Debug script to test zonotope containment after Conv2D layer evaluation
Compare with MATLAB implementation
"""

import numpy as np
import os
import sys

# Add path
sys.path.insert(0, os.path.abspath('.'))

from cora_python.nn.layers.linear.nnConv2DLayer import nnConv2DLayer
from cora_python.nn.neuralNetwork.neuralNetwork import NeuralNetwork
from cora_python.contSet.zonotope.zonotope import Zonotope

# Open output file
with open('python_zonotope_contains_output.txt', 'w') as fid:
    fid.write('=== Testing Zonotope Containment after Conv2D Layer ===\n\n')
    
    # Create the same Conv2D layer as in MATLAB test
    W = np.zeros((2, 2, 1, 2), dtype=np.float64)
    W[:, :, 0, 0] = np.array([[1, -1], [-1, 2]])  # filter 1
    W[:, :, 0, 1] = np.array([[2, 3], [-1, -2]])  # filter 2
    b = np.array([1.0, -2.0])
    
    layer = nnConv2DLayer(W, b)
    
    nn = NeuralNetwork([layer])
    n = 4
    nn.setInputSize([n, n, 1])
    
    fid.write('Layer configuration:\n')
    fid.write(f'  Input size: [{n} {n} 1]\n')
    fid.write(f'  W shape: {W.shape}\n')
    fid.write(f'  b shape: {b.shape}\n')
    fid.write('\n')
    
    # MATLAB: x = reshape(eye(n),[],1);
    # In MATLAB, this creates a column vector [n*n, 1]
    x = np.eye(n).flatten(order='F').reshape(-1, 1)  # Column-major flatten to match MATLAB
    
    fid.write('Input point x:\n')
    fid.write(f'  Shape: {x.shape}\n')
    fid.write('  First 10 values: ')
    fid.write(' '.join([f'{val:.6f}' for val in x[:10].flatten()]))
    fid.write('\n\n')
    
    # MATLAB: X = zonotope(x,0.01 * eye(n*n));
    G = 0.01 * np.eye(n * n)
    X = Zonotope(x, G)
    
    fid.write('Input zonotope X:\n')
    fid.write(f'  Center shape: {X.c.shape}\n')
    fid.write(f'  Generator shape: {X.G.shape}\n')
    fid.write(f'  Number of generators: {X.G.shape[1]}\n')
    fid.write('\n')
    
    # Evaluate zonotope through network
    fid.write('Evaluating zonotope through network...\n')
    options = {'nn': {}}
    Y = nn.evaluate(X, options)
    
    fid.write('Output zonotope Y:\n')
    fid.write(f'  Center shape: {Y.c.shape}\n')
    fid.write(f'  Generator shape: {Y.G.shape}\n')
    fid.write(f'  Number of generators: {Y.G.shape[1]}\n')
    fid.write('  Center (first 10): ')
    fid.write(' '.join([f'{val:.6f}' for val in Y.c[:10].flatten()]))
    fid.write('\n')
    fid.write('  Center (all values): ')
    fid.write(' '.join([f'{val:.6f}' for val in Y.c.flatten()]))
    fid.write('\n\n')
    
    # Evaluate point through network
    fid.write('Evaluating point through network...\n')
    y_point = nn.evaluate(x, options)
    
    fid.write('Output point y_point:\n')
    fid.write(f'  Shape: {y_point.shape}\n')
    fid.write('  First 10 values: ')
    fid.write(' '.join([f'{val:.6f}' for val in y_point[:10].flatten()]))
    fid.write('\n')
    fid.write('  All values: ')
    fid.write(' '.join([f'{val:.6f}' for val in y_point.flatten()]))
    fid.write('\n\n')
    
    # Check if center matches
    fid.write('Center comparison:\n')
    center_diff = Y.c - y_point
    max_diff = np.max(np.abs(center_diff))
    fid.write(f'  Max difference: {max_diff:.6e}\n')
    fid.write(f'  All close (tol=1e-10): {np.allclose(Y.c, y_point, atol=1e-10)}\n')
    fid.write('\n')
    
    # Check containment
    fid.write('Testing containment: Y.contains(y_point)\n')
    try:
        contains_result = Y.contains(y_point)
        fid.write(f'  Result: {contains_result}\n')
        fid.write('  Success: Containment check completed\n')
    except Exception as e:
        fid.write(f'  Error: {type(e).__name__}: {str(e)}\n')
        import traceback
        fid.write('  Traceback:\n')
        for line in traceback.format_exc().split('\n'):
            fid.write(f'    {line}\n')
    fid.write('\n')
    
    # Additional diagnostics
    fid.write('Additional diagnostics:\n')
    fid.write(f'  Y dimension: {len(Y.c)}\n')
    fid.write(f'  Y number of generators: {Y.G.shape[1]}\n')
    fid.write(f'  Is degenerate (dim > generators): {len(Y.c) > Y.G.shape[1]}\n')
    fid.write('\n')
    
    # Test with a simpler case: check if zero point is contained
    fid.write('Testing containment of zero point:\n')
    zero_point = np.zeros_like(Y.c)
    try:
        contains_zero = Y.contains(zero_point)
        fid.write(f'  Y.contains(zero): {contains_zero}\n')
    except Exception as e:
        fid.write(f'  Error: {type(e).__name__}: {str(e)}\n')
    fid.write('\n')
    
    # Compare with MATLAB values
    fid.write('=== Comparison with MATLAB ===\n')
    fid.write('MATLAB center (first 10): 4.000000 0.000000 1.000000 0.000000 4.000000 0.000000 1.000000 0.000000 4.000000 -2.000000\n')
    fid.write('Python center (first 10): ')
    fid.write(' '.join([f'{val:.6f}' for val in Y.c[:10].flatten()]))
    fid.write('\n')
    fid.write('MATLAB point (first 10): 4.000000 0.000000 1.000000 0.000000 4.000000 0.000000 1.000000 0.000000 4.000000 -2.000000\n')
    fid.write('Python point (first 10): ')
    fid.write(' '.join([f'{val:.6f}' for val in y_point[:10].flatten()]))
    fid.write('\n')

print('Results saved to python_zonotope_contains_output.txt')

