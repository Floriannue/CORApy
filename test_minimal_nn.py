#!/usr/bin/env python3
"""
Minimal test script to check neural network functionality
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

try:
    print("Testing imports...")
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    print("✓ All imports successful")
    
    print("\nTesting layer creation...")
    # Create a simple network
    W1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    b1 = np.array([[0.0], [0.0]])
    W2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b2 = np.array([[0.0], [0.0]])
    
    layers = [
        nnLinearLayer(W1, b1),
        nnReLULayer(),
        nnLinearLayer(W2, b2)
    ]
    
    nn = NeuralNetwork(layers)
    print("✓ Neural network created successfully")
    print(f"  - Number of layers: {len(nn.layers)}")
    print(f"  - Input neurons: {nn.neurons_in}")
    print(f"  - Output neurons: {nn.neurons_out}")
    
    print("\nTesting basic evaluation...")
    x = np.array([[1.0], [2.0]])
    options = {'nn': {'use_approx_error': True, 'poly_method': 'bounds'}}
    
    # Test numeric evaluation
    y = nn.evaluate(x, options)
    print("✓ Numeric evaluation successful")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {y.shape}")
    print(f"  - Output: {y.flatten()}")
    
    print("\nTesting zonotope batch evaluation preparation...")
    try:
        numGen = nn.prepareForZonoBatchEval(x, options)
        print("✓ Zonotope batch evaluation preparation successful")
        print(f"  - Number of generators: {numGen}")
    except Exception as e:
        print(f"✗ Zonotope batch evaluation preparation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting zonotope batch evaluation...")
    try:
        # Create simple zonotope data
        c = x.reshape(x.shape[0], 1, 1)  # Shape: (n, 1, 1)
        G = np.zeros((x.shape[0], numGen, 1))  # Shape: (n, numGen, 1)
        G[:, :x.shape[0], 0] = np.eye(x.shape[0])  # Identity matrix for generators
        
        yi, Gyi = nn.evaluateZonotopeBatch_(c, G, options, [0, 1, 2])
        print("✓ Zonotope batch evaluation successful")
        print(f"  - Output center shape: {yi.shape}")
        print(f"  - Output generators shape: {Gyi.shape}")
    except Exception as e:
        print(f"✗ Zonotope batch evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting verify method...")
    try:
        # Simple verification setup
        r = 0.1
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        timeout = 10.0
        verbose = False
        
        res, x_, y_ = nn.verify(nn, x, r, A, b, safeSet, options, timeout, verbose)
        print("✓ Verify method successful")
        print(f"  - Result: {res}")
        if x_ is not None:
            print(f"  - Counterexample input: {x_.flatten()}")
        if y_ is not None:
            print(f"  - Counterexample output: {y_.flatten()}")
    except Exception as e:
        print(f"✗ Verify method failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ All tests completed!")
    
except Exception as e:
    print(f"✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
