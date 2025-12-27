"""
Compare Python evaluateZonotopeBatch results with MATLAB
This script runs Python tests and compares against MATLAB if available
"""
import numpy as np
import subprocess
import sys
import os

def run_python_test():
    """Run Python test and extract results"""
    print("=" * 70)
    print("Running Python evaluateZonotopeBatch tests")
    print("=" * 70)
    
    # Import and run the test
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    
    # Test 1: Default all layers
    print("\nTest 1: evaluateZonotopeBatch_default_all_layers")
    print("-" * 70)
    
    W1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    b1 = np.array([[0.5], [1.0]])
    layer1 = nnLinearLayer(W1, b1)

    W2 = np.array([[2.0, -1.0]])
    b2 = np.array([[0.2]])
    layer2 = nnLinearLayer(W2, b2)

    nn = NeuralNetwork([layer1, layer2])

    c = np.array([[[1.0]], [[-1.0]]])
    G = np.array([[[0.1, 0.0]], [[0.0, 0.2]]])

    result_c, result_G = nn.evaluateZonotopeBatch(c, G)
    
    # Convert torch tensors to numpy for comparison
    if hasattr(result_c, 'cpu'):
        result_c = result_c.cpu().numpy()
    if hasattr(result_G, 'cpu'):
        result_G = result_G.cpu().numpy()
    
    print(f"Input c shape: {c.shape}")
    print(f"Input G shape: {G.shape}")
    print(f"Output c shape: {result_c.shape}")
    print(f"Output G shape: {result_G.shape}")
    print(f"\nOutput c:\n{result_c}")
    print(f"\nOutput G:\n{result_G}")
    
    # Compute expected values
    c_after_layer1 = np.einsum('ij,jkb->ikb', W1, c) + b1.reshape(b1.shape[0], 1, 1)
    G_after_layer1 = np.einsum('ij,jkb->ikb', W1, G)
    expected_c = np.einsum('ij,jkb->ikb', W2, c_after_layer1) + b2.reshape(b2.shape[0], 1, 1)
    expected_G = np.einsum('ij,jkb->ikb', W2, G_after_layer1)
    
    print(f"\nExpected c:\n{expected_c}")
    print(f"\nExpected G:\n{expected_G}")
    
    c_match = np.allclose(result_c, expected_c)
    G_match = np.allclose(result_G, expected_G)
    
    print(f"\nOK c matches expected: {c_match}")
    print(f"OK G matches expected: {G_match}")
    
    if not c_match:
        print(f"c difference: {np.abs(result_c - expected_c).max()}")
    if not G_match:
        print(f"G difference: {np.abs(result_G - expected_G).max()}")
    
    # Test 2: idxLayer selection
    print("\n" + "=" * 70)
    print("Test 2: evaluateZonotopeBatch_idxLayer_zero_based")
    print("-" * 70)
    
    W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b1 = np.zeros((2, 1))
    layer1 = nnLinearLayer(W1, b1)

    W2 = np.array([[2.0, 3.0]])
    b2 = np.array([[1.0]])
    layer2 = nnLinearLayer(W2, b2)

    nn = NeuralNetwork([layer1, layer2])

    c = np.array([[[2.0]], [[3.0]]])
    G = np.array([[[0.5]], [[0.2]]])

    result_c, result_G = nn.evaluateZonotopeBatch(c, G, idxLayer=[0])
    
    # Convert torch tensors to numpy for comparison
    if hasattr(result_c, 'cpu'):
        result_c = result_c.cpu().numpy()
    if hasattr(result_G, 'cpu'):
        result_G = result_G.cpu().numpy()

    expected_c = np.einsum('ij,jkb->ikb', W1, c) + b1.reshape(b1.shape[0], 1, 1)
    expected_G = np.einsum('ij,jkb->ikb', W1, G)
    
    print(f"Output c:\n{result_c}")
    print(f"Expected c:\n{expected_c}")
    print(f"Output G:\n{result_G}")
    print(f"Expected G:\n{expected_G}")
    
    c_match = np.allclose(result_c, expected_c)
    G_match = np.allclose(result_G, expected_G)
    
    print(f"\nOK: c matches expected: {c_match}")
    print(f"OK: G matches expected: {G_match}")
    
    # Test 3: Backprop storage
    print("\n" + "=" * 70)
    print("Test 3: evaluateZonotopeBatch_stores_inputs_for_backprop")
    print("-" * 70)
    
    W = np.eye(2)
    b = np.zeros((2, 1))
    layer = nnLinearLayer(W, b)
    nn = NeuralNetwork([layer])

    c = np.array([[[1.0]], [[-1.0]]])
    G = np.array([[[0.2]], [[0.3]]])

    options = {'nn': {'train': {'backprop': True}}}
    nn.evaluateZonotopeBatch(c, G, options=options)

    has_store = 'store' in layer.backprop
    has_inc = 'inc' in layer.backprop['store'] if has_store else False
    has_inG = 'inG' in layer.backprop['store'] if has_store else False
    
    print(f"OK: backprop.store exists: {has_store}")
    print(f"OK: backprop.store.inc exists: {has_inc}")
    print(f"OK: backprop.store.inG exists: {has_inG}")
    
    if has_inc and has_inG:
        inc_match = np.allclose(layer.backprop['store']['inc'], c)
        inG_match = np.allclose(layer.backprop['store']['inG'], G)
        print(f"OK: inc matches input c: {inc_match}")
        print(f"OK: inG matches input G: {inG_match}")
    
    print("\n" + "=" * 70)
    print("Python tests completed!")
    print("=" * 70)
    
    # Try to run MATLAB comparison if available
    print("\nAttempting to run MATLAB comparison...")
    matlab_script = "debug_matlab_evaluateZonotopeBatch.m"
    if os.path.exists(matlab_script):
        try:
            result = subprocess.run(
                ['matlab', '-batch', f"run('{matlab_script}')"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print("MATLAB script executed successfully!")
                print(result.stdout)
                if os.path.exists('matlab_evaluateZonotopeBatch_results.mat'):
                    print("\nMATLAB results saved to matlab_evaluateZonotopeBatch_results.mat")
                    print("You can load this in MATLAB to compare with Python results.")
            else:
                print(f"MATLAB execution failed: {result.stderr}")
        except FileNotFoundError:
            print("MATLAB not found in PATH. Skipping MATLAB comparison.")
        except subprocess.TimeoutExpired:
            print("MATLAB execution timed out.")
        except Exception as e:
            print(f"Error running MATLAB: {e}")
    else:
        print(f"MATLAB script {matlab_script} not found.")

if __name__ == "__main__":
    run_python_test()

