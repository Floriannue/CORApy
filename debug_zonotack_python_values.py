"""
Debug script to capture actual values from Python zonotack attack
This will run the test and print intermediate values
"""
import sys
import numpy as np
sys.path.insert(0, 'cora_python')

from nn.neuralNetwork.neuralNetwork import neuralNetwork
from nn.layers.linear.nnLinearLayer import nnLinearLayer
from nn.layers.activation.nnReLULayer import nnReLULayer

# Create the neural network (same as test)
layers = [
    nnLinearLayer(
        np.array([[0.6294, 0.2647], [0.8116, -0.8049], [-0.7460, -0.4430], [0.8268, 0.0938]]),
        np.array([[0.9150], [0.9298], [-0.6848], [0.9412]])
    ),
    nnReLULayer(),
    nnLinearLayer(
        np.array([[0.9143, -0.1565, 0.3115, 0.3575], [-0.0292, 0.8315, -0.9286, 0.5155], [0.6006, 0.5844, 0.6983, 0.4863], [-0.7162, 0.9190, 0.8680, -0.2155]]),
        np.array([[0.3110], [-0.6576], [0.4121], [-0.9363]])
    ),
    nnReLULayer(),
    nnLinearLayer(
        np.array([[-0.4462, -0.8057, 0.3897, 0.9004], [-0.9077, 0.6469, -0.3658, -0.9311]]),
        np.array([[-0.1225], [-0.2369]])
    ),
]
nn = neuralNetwork(layers)

# Test parameters
x = np.array([[0.0], [0.0]])
r = np.array([[1.0], [1.0]])
A = np.array([[-1.0, 1.0]])
bunsafe = -1.27
safeSet = False

# Options
options = {
    'nn': {
        'use_approx_error': True,
        'poly_method': 'bounds',
        'train': {
            'backprop': False,
            'mini_batch_size': 512,
            'num_init_gens': 2
        },
        'interval_center': False,
        'falsification_method': 'zonotack',
        'refinement_method': 'zonotack'
    }
}

print("=== Python Zonotack Attack Debug ===\n")
print(f"Configuration:")
print(f"  x: {x.flatten()}")
print(f"  r: {r.flatten()}")
print(f"  A: {A}")
print(f"  b: {bunsafe}")
print(f"  numInitGens: {options['nn']['train']['num_init_gens']}")
print()

# We need to patch verify.py to print intermediate values
# For now, let's just run verify and see what we get
try:
    result = nn.verify(x, r, A, bunsafe, safeSet, options, timeout=2, verbose=False)
    print(f"Result: {result}")
    if hasattr(result, 'x_') and result.x_ is not None:
        print(f"Counterexample x_: {result.x_.flatten()}")
        print(f"Counterexample y_: {result.y_.flatten() if hasattr(result, 'y_') and result.y_ is not None else 'N/A'}")
        
        # Verify bounds
        in_bounds = np.all(result.x_ >= x - r) and np.all(result.x_ <= x + r)
        print(f"x_ in bounds: {in_bounds}")
        if not in_bounds:
            print(f"  Lower: {(x - r).flatten()}")
            print(f"  Upper: {(x + r).flatten()}")
            print(f"  x_: {result.x_.flatten()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Need to add debug output to verify.py ===")
print("We need to print Gxi, beta, delta, and zi values during zonotack attack")


