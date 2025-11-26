"""
Test file for NeuralNetwork.verify method

This file tests the verification method of the NeuralNetwork class.
Includes tests matching cora/unitTests/nn/neuralNetwork/test_nn_neuralNetwork_verify.m exactly.
"""

import pytest
import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions


class TestNeuralNetworkVerify:
    """Test class for NeuralNetwork.verify method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create a simple neural network
        W1 = np.array([[1, 2], [3, 4]])
        b1 = np.array([[0], [0]])
        W2 = np.array([[1, 0], [0, 1]])
        b2 = np.array([[0], [0]])
        
        layers = [
            nnLinearLayer(W1, b1),
            nnReLULayer(),
            nnLinearLayer(W2, b2)
        ]
        
        self.nn = NeuralNetwork(layers)
    
    def test_verify_basic(self):
        """Test basic verification"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options)
        
        # Should return result string and optional counterexamples
        assert isinstance(res, str)
        assert res in ['VERIFIED', 'COUNTEREXAMPLE', 'UNKNOWN']
    
    def test_verify_with_timeout(self):
        """Test verification with timeout"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        timeout = 0.001  # Very short timeout
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options, timeout)
        
        # Should return UNKNOWN due to timeout, but if it finds a counterexample quickly, that's also valid
        assert res in ['UNKNOWN', 'COUNTEREXAMPLE']
        # If counterexample found before timeout, x_ and y_ will not be None
        if res == 'UNKNOWN':
            assert x_ is None or x_.size == 0
            assert y_ is None or y_.size == 0
    
    def test_verify_with_options(self):
        """Test verification with various options"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {
            'nn': {
                'train': {
                    'mini_batch_size': 16,
                    'use_gpu': False
                },
                'interval_center': False
            }
        }
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options)
        
        # Should work with options
        assert isinstance(res, str)
    
    def test_verify_safe_set_true(self):
        """Test verification with safeSet=True"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options)
        
        # Should work with safeSet=True
        assert isinstance(res, str)
    
    def test_verify_safe_set_false(self):
        """Test verification with safeSet=False"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = False
        options = {}
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options)
        
        # Should work with safeSet=False
        assert isinstance(res, str)
    
    def test_verify_none_options(self):
        """Test verification with None options"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options=None)
        
        # Should work with default options
        assert isinstance(res, str)
    
    def test_verify_none_timeout(self):
        """Test verification with None timeout (should use default)"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options, timeout=None)
        
        # Should work with default timeout
        assert isinstance(res, str)
    
    def test_verify_verbose(self):
        """Test verification with verbose output"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        
        # Should not raise exception with verbose=True
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options, verbose=True)
        
        assert isinstance(res, str)
    
    def test_verify_different_network(self):
        """Test verification with different network instance"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        
        # Create another network instance
        other_nn = NeuralNetwork([])
        
        res, x_, y_ = other_nn.verify(x, r, A, b, safeSet, options)
        
        # Should work with different network
        assert isinstance(res, str)
    
    def test_verify_aux_pop(self):
        """Test _aux_pop helper method by importing it directly"""
        from cora_python.nn.neuralNetwork.verify import _aux_pop
        
        xs = np.array([[1, 2, 3], [4, 5, 6]])
        rs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        bs = 2
        
        xi, ri, xs_new, rs_new = _aux_pop(xs, rs, bs)
        
        # Check shapes
        assert xi.shape == (2, 2)
        assert ri.shape == (2, 2)
        assert xs_new.shape == (2, 1)
        assert rs_new.shape == (2, 1)
    
    def test_verify_aux_split(self):
        """Test _aux_split helper method by importing it directly"""
        from cora_python.nn.neuralNetwork.verify import _aux_split
        
        xi = np.array([[1, 2], [3, 4]])
        ri = np.array([[0.1, 0.2], [0.3, 0.4]])
        sens = np.array([[0.5, 0.6], [0.7, 0.8]])
        nSplits = 3
        nDims = 1
        
        xis, ris = _aux_split(xi, ri, sens, nSplits, nDims)
        
        # Check shapes
        assert xis.shape[0] == 2
        assert ris.shape[0] == 2
        assert xis.shape[1] == 6  # 2 * 3 splits
        assert ris.shape[1] == 6  # 2 * 3 splits


def test_nn_neuralNetwork_verify_matlab_exact():
    """
    Test neuralNetwork.verify with hardcoded values matching MATLAB test exactly.
    
    This test matches cora/unitTests/nn/neuralNetwork/test_nn_neuralNetwork_verify.m exactly.
    """
    # Reset the random number generator to match MATLAB
    np.random.seed(0)  # MATLAB's 'default' seed
    
    # Create the neural network. The weights are from a randomly generated
    # neural network (matching MATLAB test exactly):
    # rng('default');
    # nn = neuralNetwork.generateRandom(NrInputs=2,NrOutputs=2, ...
    #     ActivationFun='relu',NrLayers=3,NrHiddenNeurons=4);
    # nn.layers(end) = [];
    
    layers = [
        nnLinearLayer(
            np.array([[0.6294, 0.2647], [0.8116, -0.8049], [-0.7460, -0.4430], [0.8268, 0.0938]]),
            np.array([[0.9150], [0.9298], [-0.6848], [0.9412]])
        ),
        nnReLULayer(),
        nnLinearLayer(
            np.array([[0.9143, -0.1565, 0.3115, 0.3575], 
                      [-0.0292, 0.8315, -0.9286, 0.5155], 
                      [0.6006, 0.5844, 0.6983, 0.4863], 
                      [-0.7162, 0.9190, 0.8680, -0.2155]]),
            np.array([[0.3110], [-0.6576], [0.4121], [-0.9363]])
        ),
        nnReLULayer(),
        nnLinearLayer(
            np.array([[-0.4462, -0.8057, 0.3897, 0.9004], 
                      [-0.9077, 0.6469, -0.3658, -0.9311]]),
            np.array([[-0.1225], [-0.2369]])
        ),
    ]
    nn = NeuralNetwork(layers)
    
    # Specify initial set.
    x = np.array([[0], [0]])  # center
    r = np.array([[1], [1]])  # radius
    
    # Specify unsafe set specification.
    A = np.array([[-1, 1]])  # Shape: (1, 2)
    bsafe = -2.27
    bunsafe = -1.27
    safeSet = False
    
    # Verbose verification output.
    verbose = True
    # Set a timeout of 2s.
    timeout = 2
    
    # Create evaluation options.
    options = {}
    options['nn'] = {
        'use_approx_error': True,
        'poly_method': 'bounds',
        'train': {
            'backprop': False,
            'mini_batch_size': 512
        }
    }
    # Set default training parameters
    options = validateNNoptions(options, True)
    options['nn']['interval_center'] = False
    
    # Set the falsification method: {'fgsm','center','zonotack'}.
    options['nn']['falsification_method'] = 'zonotack'
    # Set the input set refinement method: {'naive','zonotack'}.
    options['nn']['refinement_method'] = 'zonotack'
    
    # Do verification - should return VERIFIED
    # MATLAB: [res,x_,y_] = nn.verify(x,r,A,bsafe,safeSet,options,timeout,verbose,[1:2; 1:2]);
    # MATLAB: assert(strcmp(res.str,'VERIFIED') & isempty(x_) & isempty(y_));
    # plotDims = [1:2; 1:2] means plotDims = [[1, 2], [1, 2]] for input and output
    plotDims = [[1, 2], [1, 2]]
    res, x_, y_ = nn.verify(x, r, A, bsafe, safeSet, options, timeout, verbose, plotDims, False)
    
    # Match MATLAB assertion: strcmp(res.str,'VERIFIED') & isempty(x_) & isempty(y_)
    # In Python, res is a string (not a struct), so check res == 'VERIFIED'
    assert res == 'VERIFIED', f"Expected 'VERIFIED', got '{res}'"
    # MATLAB: isempty(x_) means x_ is empty array [] or None
    # In Python, check if x_ is None or has size 0
    assert x_ is None or (hasattr(x_, 'size') and x_.size == 0), f"Expected empty x_, got shape {x_.shape if x_ is not None else None}"
    assert y_ is None or (hasattr(y_, 'size') and y_.size == 0), f"Expected empty y_, got shape {y_.shape if y_ is not None else None}"
    
    # Find counterexample - should return COUNTEREXAMPLE
    # MATLAB: [res,x_,y_] = nn.verify(x,r,A,bunsafe,safeSet,options,timeout,verbose,[1:2; 1:2]);
    # MATLAB: assert(strcmp(res.str,'COUNTEREXAMPLE') & ~isempty(x_) & ~isempty(y_) ... & aux_checkCounterexample(...));
    res, x_, y_ = nn.verify(x, r, A, bunsafe, safeSet, options, timeout, verbose, plotDims, False)
    
    # Match MATLAB assertion: strcmp(res.str,'COUNTEREXAMPLE') & ~isempty(x_) & ~isempty(y_) & aux_checkCounterexample(...)
    assert res == 'COUNTEREXAMPLE', f"Expected 'COUNTEREXAMPLE', got '{res}'"
    # MATLAB: ~isempty(x_) means x_ is not empty
    assert x_ is not None and x_.size > 0, f"Expected non-empty x_, got {x_}"
    assert y_ is not None and y_.size > 0, f"Expected non-empty y_, got {y_}"
    assert aux_checkCounterexample(nn, A, bunsafe, safeSet, x_, y_)


def aux_checkCounterexample(nn, A, b, safeSet, x_, y_):
    """
    Check if counterexample is valid (matching MATLAB aux_checkCounterexample)
    
    Args:
        nn: Neural network
        A: Specification matrix
        b: Specification bound
        safeSet: Whether this is a safe set specification
        x_: Counterexample input
        y_: Counterexample output
        
    Returns:
        True if counterexample is valid
    """
    # Compute output of the neural network.
    yi = nn.evaluate(x_)
    
    # Check if output matches.
    res = np.allclose(y_, yi, atol=1e-7)
    
    # Check if output violates the specification.
    # MATLAB: if safeSet
    #     violates = any(A*yi >= b,1);
    # else
    #     violates = all(A*yi <= b,1);
    # end
    if safeSet:
        # For safe set: violation means any(A*yi >= b)
        violates = np.any(A @ yi >= b)
    else:
        # For unsafe set: violation means all(A*yi <= b)
        # Note: This matches MATLAB's logic - if all constraints are satisfied (all A*y <= b),
        # it means the property is NOT satisfied (it's unsafe), so we found a counterexample.
        violates = np.all(A @ yi <= b)
    
    assert res and violates
    return True
