# Debug script to test Python explain method
import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer

print("Python Results:")

# construct network
W1 = np.array([[-9, -8, -7], [10, -6, 0], [-6, 2, 5], [4, 4, -8], [-5, -8, 2],
                [0, 6, 2], [-7, 10, -2], [0, 8, 6], [1, -3, -2], [3, 9, 2]])
W2 = np.array([[3, 6, -5, 3, -6, 2, 6, 2, -4, 8],
                [4, 1, 7, -3, -4, 4, 2, 0, 2, -1],
                [-3, 9, 1, 5, 10, 9, 1, 4, -6, -7]])

nn = NeuralNetwork([
    nnLinearLayer(W1),
    nnSigmoidLayer(),
    nnLinearLayer(W2)
])

# construct input
x = np.array([[1], [2], [3]])
label = 1

# compute explanation
verbose = False
epsilon = 0.2

# method: standard
method = 'standard'
print(f'Python Results for method: {method}')
result_standard = nn.explain(x, label, epsilon, InputSize=[3, 1, 1],
                           Method=method, Verbose=verbose)

print(f'result_standard type: {type(result_standard)}')
print(f'result_standard: {result_standard}')

# method: abstract+refine
method = 'abstract+refine'
print(f'\nPython Results for method: {method}')
result_abstract = nn.explain(x, label, epsilon, InputSize=[3, 1, 1],
                           Method=method, Verbose=verbose)

print(f'result_abstract type: {type(result_abstract)}')
print(f'result_abstract: {result_abstract}')

# Test with simple network
print('\n=== Simple Network Test ===')
W1_simple = np.array([[1, 2], [3, 4]])
W2_simple = np.array([[1, 0], [0, 1]])
nn_simple = NeuralNetwork([
    nnLinearLayer(W1_simple),
    nnSigmoidLayer(),
    nnLinearLayer(W2_simple)
])

x_simple = np.array([[1], [2]])
label_simple = 0
epsilon_simple = 0.2

result_simple = nn_simple.explain(x_simple, label_simple, epsilon_simple, 
                                InputSize=[2, 1, 1], Method='standard', Verbose=False)

print(f'Simple network result type: {type(result_simple)}')
print(f'Simple network result: {result_simple}')
