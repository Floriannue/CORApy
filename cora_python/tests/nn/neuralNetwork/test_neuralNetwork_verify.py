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
        from cora_python.nn.neuralNetwork.verify_helpers import _aux_pop
        
        xs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        rs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        nrXs = np.zeros((0, 3), dtype=np.float32)  # Empty neuron split indices
        bs = 2
        options = {'nn': {'verify_dequeue_type': 'front'}}
        
        xi, ri, nrXi, xs_new, rs_new, nrXs_new, qIdx = _aux_pop(xs, rs, nrXs, bs, options)
        
        # Check shapes
        assert xi.shape == (2, 2)
        assert ri.shape == (2, 2)
        assert nrXi.shape == (0, 2)  # Empty neuron split indices
        assert xs_new.shape == (2, 1)
        assert rs_new.shape == (2, 1)
        assert nrXs_new.shape == (0, 1)
        assert qIdx.shape == (2,)
        assert np.all(qIdx == np.array([1, 2]))  # 1-based indices


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
    Check if counterexample is valid (matching MATLAB aux_checkCounterexample exactly)
    
    MATLAB signature:
    function res = aux_checkCounterexample(nn,A,b,safeSet,x_,y_)
    
    Args:
        nn: Neural network
        A: Specification matrix
        b: Specification bound (scalar or array)
        safeSet: Whether this is a safe set specification
        x_: Counterexample input
        y_: Counterexample output
        
    Returns:
        True if counterexample is valid (raises AssertionError if invalid)
    """
    # Compute output of the neural network.
    # MATLAB: yi = nn.evaluate(x_);
    yi = nn.evaluate(x_)
    
    # Check if output matches.
    # MATLAB: res = all(abs(y_ - yi) <= 1e-7,'all');
    res = np.all(np.abs(y_ - yi) <= 1e-7)
    
    # Check if output violates the specification.
    # MATLAB: if safeSet
    #     violates = any(A*yi >= b,1);
    # else
    #     violates = all(A*yi <= b,1);
    # end
    if safeSet:
        # For safe set: violation means any(A*yi >= b)
        # MATLAB: violates = any(A*yi >= b,1);
        # A*yi has shape (num_constraints, batch_size), check along axis=0 (constraints)
        violates = np.any(A @ yi >= b)
    else:
        # For unsafe set: violation means all(A*yi <= b)
        # MATLAB: violates = all(A*yi <= b,1);
        # A*yi has shape (num_constraints, batch_size), check along axis=0 (constraints)
        violates = np.all(A @ yi <= b)
    
    # MATLAB: assert(res & violates);
    assert res and violates, f"Counterexample check failed: res={res}, violates={violates}"
    return True


def aux_readNetworkAndOptions(modelPath: str, vnnlibPath: str):
    """
    Read network and options from ONNX and VNNLIB files (MATLAB aux_readNetworkAndOptions equivalent)
    
    MATLAB signature:
    function [nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(modelPath,vnnlibPath)
    
    Args:
        modelPath: Path to ONNX model file
        vnnlibPath: Path to VNNLIB specification file
        
    Returns:
        Tuple of (nn, options, x, r, A, b, safeSet)
    """
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    # Create evaluation options.
    # MATLAB: options.nn = struct('use_approx_error',true,'poly_method','bounds','train',struct('backprop',false,'mini_batch_size',2^8));
    options = {}
    options['nn'] = {
        'use_approx_error': True,
        'poly_method': 'bounds',
        'train': {
            'backprop': False,
            'mini_batch_size': 2**8  # 2^8 = 256
        }
    }
    # Set default training parameters
    # MATLAB: options = nnHelper.validateNNoptions(options,true);
    options = validateNNoptions(options, True)
    options['nn']['interval_center'] = False
    
    # Read the neural network.
    # MATLAB: nn = neuralNetwork.readONNXNetwork(modelPath,false,'BSSC');
    nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BSSC')
    
    # Read the input set and specification.
    # MATLAB: [X0,specs] = vnnlib2cora(vnnlibPath);
    X0, specs = vnnlib2cora(vnnlibPath)
    
    # Extract input set.
    # MATLAB: x = 1/2*(X0{1}.sup + X0{1}.inf);
    # MATLAB: r = 1/2*(X0{1}.sup - X0{1}.inf);
    x = 0.5 * (X0[0].sup + X0[0].inf)
    r = 0.5 * (X0[0].sup - X0[0].inf)
    
    # Extract specification.
    # MATLAB: if isa(specs.set,'halfspace')
    #     A = specs.set.c';
    #     b = specs.set.d;
    # else
    #     A = specs.set.A;
    #     b = specs.set.b;
    # end
    # In Python, halfspace is deprecated and represented as a polytope
    # Check if polytope represents a halfspace using representsa_
    from cora_python.contSet.polytope.representsa_ import representsa_
    isHalfspace = representsa_(specs.set, 'halfspace')
    
    if isHalfspace:
        # Halfspace case: c' * x <= d
        # MATLAB: A = specs.set.c';
        # MATLAB: b = specs.set.d;
        # In Python, a halfspace polytope has A (1, n) and b (1, 1)
        # Extract as c' and d to match MATLAB
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
    
    return nn, options, x, r, A, b, safeSet


def testnn_neuralNetwork_verify():
    """
    Test neuralNetwork.verify function with ACASXU models (MATLAB testnn_neuralNetwork_verify equivalent)
    
    This test matches cora/unitTests/nn/neuralNetwork/testnn_neuralNetwork_verify.m exactly.
    Uses specs from the ACASXU benchmark: prop_1, prop_2.
    """
    import os
    import pytest
    from cora_python.g.macros.CORAROOT import CORAROOT
    
    # We use the specs from the acasxu benchmark: prop_1, prop_2, prop_3, and prop_5.
    
    # Toggle verbose verification output.
    verbose = True
    
    # Specify the model path.
    cora_root = CORAROOT()
    model1Path = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx')
    model2Path = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_5_3_batch_2000.onnx')
    prop1Filename = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'prop_1.vnnlib')
    prop2Filename = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'prop_2.vnnlib')
    
    # Check if files exist, skip test if they don't
    if not os.path.isfile(model1Path):
        pytest.skip(f"ACASXU model file not found: {model1Path}")
    if not os.path.isfile(model2Path):
        pytest.skip(f"ACASXU model file not found: {model2Path}")
    if not os.path.isfile(prop1Filename):
        pytest.skip(f"VNNLIB file not found: {prop1Filename}")
    if not os.path.isfile(prop2Filename):
        pytest.skip(f"VNNLIB file not found: {prop2Filename}")
    
    # Set a timeout of 10s (MATLAB: timeout = 10;)
    timeout = 10
    
    # First test case: prop_1.vnnlib ------------------------------------------
    # MATLAB: [nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(model1Path,prop1Filename);
    nn, options, x, r, A, b, safeSet = aux_readNetworkAndOptions(model1Path, prop1Filename)
    
    # Test 'naive'-splitting and 'fgsm'-falsification.
    # MATLAB: options.nn.falsification_method = 'fgsm';
    # MATLAB: options.nn.refinement_method = 'naive';
    options['nn']['falsification_method'] = 'fgsm'
    options['nn']['refinement_method'] = 'naive'
    
    # Do verification.
    # MATLAB: [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);
    # MATLAB: assert(~strcmp(verifRes.str,'COUNTEREXAMPLE') & isempty(x_) & isempty(y_));
    verifRes, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    assert verifRes != 'COUNTEREXAMPLE', f"Expected not COUNTEREXAMPLE, got '{verifRes}'"
    assert x_ is None or (hasattr(x_, 'size') and x_.size == 0), f"Expected empty x_, got {x_}"
    assert y_ is None or (hasattr(y_, 'size') and y_.size == 0), f"Expected empty y_, got {y_}"
    
    # Test 'zonotack' implementation with restricted number of generators.
    # MATLAB: options.nn.falsification_method = 'zonotack';
    # MATLAB: options.nn.refinement_method = 'zonotack';
    options['nn']['falsification_method'] = 'zonotack'
    options['nn']['refinement_method'] = 'zonotack'
    # Specify parameters.
    # MATLAB: options.nn.num_splits = 2;
    # MATLAB: options.nn.num_dimensions = 2;
    # MATLAB: options.nn.num_neuron_splits = 1;
    options['nn']['num_splits'] = 2
    options['nn']['num_dimensions'] = 2
    options['nn']['num_neuron_splits'] = 1
    # Restrict the number of input generators.
    # MATLAB: options.nn.train.num_init_gens = 5;
    options['nn']['train']['num_init_gens'] = 5
    # Restrict the number of approximation error generators per layer.
    # MATLAB: options.nn.train.num_approx_err = 50;
    options['nn']['train']['num_approx_err'] = 50
    # MATLAB: options.nn.approx_error_order = 'sensitivity*length';
    options['nn']['approx_error_order'] = 'sensitivity*length'
    # Add relu tightening constraints.
    # MATLAB: options.nn.num_relu_tighten_constraints = inf;
    options['nn']['num_relu_tighten_constraints'] = np.inf
    # Do verification.
    # MATLAB: [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);
    # MATLAB: assert(~strcmp(verifRes.str,'COUNTEREXAMPLE') & isempty(x_) & isempty(y_));
    verifRes, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    assert verifRes != 'COUNTEREXAMPLE', f"Expected not COUNTEREXAMPLE, got '{verifRes}'"
    assert x_ is None or (hasattr(x_, 'size') and x_.size == 0), f"Expected empty x_, got {x_}"
    assert y_ is None or (hasattr(y_, 'size') and y_.size == 0), f"Expected empty y_, got {y_}"
    
    # Second test case: prop_2.vnnlib -----------------------------------------
    # MATLAB: [nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(model1Path,prop2Filename);
    nn, options, x, r, A, b, safeSet = aux_readNetworkAndOptions(model1Path, prop2Filename)
    
    # Test 'naive'-splitting and 'fgsm'-falsification.
    # MATLAB: options.nn.falsification_method = 'fgsm';
    # MATLAB: options.nn.refinement_method = 'naive';
    options['nn']['falsification_method'] = 'fgsm'
    options['nn']['refinement_method'] = 'naive'
    
    # Do verification.
    # MATLAB: [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);
    # MATLAB: assert(~strcmp(verifRes.str,'VERIFIED'));
    verifRes, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    assert verifRes != 'VERIFIED', f"Expected not VERIFIED, got '{verifRes}'"
    # MATLAB: if strcmp(verifRes.str,'COUNTEREXAMPLE')
    #     assert(~isempty(x_) & ~isempty(y_) & aux_checkCounterexample(nn,A,b,safeSet,x_,y_));
    # end
    if verifRes == 'COUNTEREXAMPLE':
        assert x_ is not None and x_.size > 0, f"Expected non-empty x_, got {x_}"
        assert y_ is not None and y_.size > 0, f"Expected non-empty y_, got {y_}"
        assert aux_checkCounterexample(nn, A, b, safeSet, x_, y_)
    
    # Test 'zonotack' implementation with restricted number of generators.
    # MATLAB: options.nn.falsification_method = 'zonotack';
    # MATLAB: options.nn.refinement_method = 'zonotack-layerwise';
    options['nn']['falsification_method'] = 'zonotack'
    options['nn']['refinement_method'] = 'zonotack-layerwise'
    # Specify parameters.
    # MATLAB: options.nn.num_splits = 3;
    # MATLAB: options.nn.num_dimensions = 1;
    # MATLAB: options.nn.num_neuron_splits = 0;
    options['nn']['num_splits'] = 3
    options['nn']['num_dimensions'] = 1
    options['nn']['num_neuron_splits'] = 0
    # Restrict the number of input generators.
    # MATLAB: options.nn.train.num_init_gens = 5;
    options['nn']['train']['num_init_gens'] = 5
    # Restrict the number of approximation error generators per layer.
    # MATLAB: options.nn.train.num_approx_err = 25;
    options['nn']['train']['num_approx_err'] = 25
    # MATLAB: options.nn.approx_error_order = 'length';
    options['nn']['approx_error_order'] = 'length'
    # Add relu tightening constraints.
    # MATLAB: options.nn.num_relu_tighten_constraints = 3;
    options['nn']['num_relu_tighten_constraints'] = 3
    # Do verification.
    # MATLAB: [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);
    # MATLAB: assert(~strcmp(verifRes.str,'VERIFIED'));
    verifRes, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    assert verifRes != 'VERIFIED', f"Expected not VERIFIED, got '{verifRes}'"
    # MATLAB: if strcmp(verifRes.str,'COUNTEREXAMPLE')
    #     assert(~isempty(x_) & ~isempty(y_) & aux_checkCounterexample(nn,A,b,safeSet,x_,y_));
    # end
    if verifRes == 'COUNTEREXAMPLE':
        assert x_ is not None and x_.size > 0, f"Expected non-empty x_, got {x_}"
        assert y_ is not None and y_.size > 0, f"Expected non-empty y_, got {y_}"
        assert aux_checkCounterexample(nn, A, b, safeSet, x_, y_)
    
    # Test 'zonotack' implementation with restricted number of generators.
    # MATLAB: options.nn.falsification_method = 'zonotack';
    # MATLAB: options.nn.refinement_method = 'zonotack-layerwise';
    options['nn']['falsification_method'] = 'zonotack'
    options['nn']['refinement_method'] = 'zonotack-layerwise'
    # Specify parameters.
    # MATLAB: options.nn.num_splits = 2;
    # MATLAB: options.nn.num_dimensions = 1;
    # MATLAB: options.nn.num_neuron_splits = 0;
    options['nn']['num_splits'] = 2
    options['nn']['num_dimensions'] = 1
    options['nn']['num_neuron_splits'] = 0
    # Add relu tightening constraints.
    # MATLAB: options.nn.num_relu_tighten_constraints = 100;
    options['nn']['num_relu_tighten_constraints'] = 100
    # Do verification.
    # MATLAB: [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);
    # MATLAB: assert(~strcmp(verifRes.str,'VERIFIED'));
    verifRes, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    assert verifRes != 'VERIFIED', f"Expected not VERIFIED, got '{verifRes}'"
    # MATLAB: if strcmp(verifRes.str,'COUNTEREXAMPLE')
    #     assert(~isempty(x_) & ~isempty(y_) & aux_checkCounterexample(nn,A,b,safeSet,x_,y_));
    # end
    if verifRes == 'COUNTEREXAMPLE':
        assert x_ is not None and x_.size > 0, f"Expected non-empty x_, got {x_}"
        assert y_ is not None and y_.size > 0, f"Expected non-empty y_, got {y_}"
        assert aux_checkCounterexample(nn, A, b, safeSet, x_, y_)
    
    # Third test case with other model: prop_2.vnnlib -------------------------
    # MATLAB: [nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(model2Path,prop2Filename);
    nn, options, x, r, A, b, safeSet = aux_readNetworkAndOptions(model2Path, prop2Filename)
    
    # MATLAB: options.nn.falsification_method = 'zonotack';
    # MATLAB: options.nn.refinement_method = 'zonotack';
    options['nn']['falsification_method'] = 'zonotack'
    options['nn']['refinement_method'] = 'zonotack'
    # Specify parameters.
    # MATLAB: options.nn.num_splits = 2;
    # MATLAB: options.nn.num_dimensions = 0;
    # MATLAB: options.nn.num_neuron_splits = 1;
    options['nn']['num_splits'] = 2
    options['nn']['num_dimensions'] = 0
    options['nn']['num_neuron_splits'] = 1
    # MATLAB: options.nn.add_orth_neuron_splits = true;
    options['nn']['add_orth_neuron_splits'] = True
    # Add relu tightening constraints.
    # MATLAB: options.nn.num_relu_constraints = 15;
    options['nn']['num_relu_constraints'] = 15
    # Do verification.
    # MATLAB: [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);
    # MATLAB: assert(~strcmp(verifRes.str,'VERIFIED'));
    verifRes, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    assert verifRes != 'VERIFIED', f"Expected not VERIFIED, got '{verifRes}'"
    # MATLAB: if strcmp(verifRes.str,'COUNTEREXAMPLE')
    #     assert(~isempty(x_) & ~isempty(y_) & aux_checkCounterexample(nn,A,b,safeSet,x_,y_));
    # end
    if verifRes == 'COUNTEREXAMPLE':
        assert x_ is not None and x_.size > 0, f"Expected non-empty x_, got {x_}"
        assert y_ is not None and y_.size > 0, f"Expected non-empty y_, got {y_}"
        assert aux_checkCounterexample(nn, A, b, safeSet, x_, y_)
    
    # Fourth test case with other model: prop_2.vnnlib -------------------------
    # MATLAB: [nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(model2Path,prop2Filename);
    nn, options, x, r, A, b, safeSet = aux_readNetworkAndOptions(model2Path, prop2Filename)
    
    # MATLAB: options.nn.falsification_method = 'zonotack';
    # MATLAB: options.nn.refinement_method = 'zonotack';
    options['nn']['falsification_method'] = 'zonotack'
    options['nn']['refinement_method'] = 'zonotack'
    # Specify parameters.
    # MATLAB: options.nn.num_splits = 2;
    # MATLAB: options.nn.num_dimensions = 1;
    # MATLAB: options.nn.num_neuron_splits = 1;
    options['nn']['num_splits'] = 2
    options['nn']['num_dimensions'] = 1
    options['nn']['num_neuron_splits'] = 1
    # MATLAB: options.nn.input_xor_neuron_splitting = true;
    options['nn']['input_xor_neuron_splitting'] = True
    # Add relu tightening constraints.
    # MATLAB: options.nn.num_relu_constraints = inf;
    options['nn']['num_relu_constraints'] = np.inf
    # Do verification.
    # MATLAB: [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);
    # MATLAB: assert(~strcmp(verifRes.str,'VERIFIED'));
    verifRes, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    assert verifRes != 'VERIFIED', f"Expected not VERIFIED, got '{verifRes}'"
    # MATLAB: if strcmp(verifRes.str,'COUNTEREXAMPLE')
    #     assert(~isempty(x_) & ~isempty(y_) & aux_checkCounterexample(nn,A,b,safeSet,x_,y_));
    # end
    if verifRes == 'COUNTEREXAMPLE':
        assert x_ is not None and x_.size > 0, f"Expected non-empty x_, got {x_}"
        assert y_ is not None and y_.size > 0, f"Expected non-empty y_, got {y_}"
        assert aux_checkCounterexample(nn, A, b, safeSet, x_, y_)
