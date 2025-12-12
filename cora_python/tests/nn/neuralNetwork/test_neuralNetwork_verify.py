"""
Test file for NeuralNetwork.verify method

This file tests the verification method of the NeuralNetwork class.
Matches cora_matlab/unitTests/nn/testnn_verify.m exactly.
"""

import pytest
import numpy as np
import os
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
from cora_python.g.macros.CORAROOT import CORAROOT


def testnn_neuralNetwork_verify():
    """
    Test neuralNetwork.verify function (MATLAB testnn_neuralNetwork_verify equivalent)
    
    This test matches cora_matlab/unitTests/nn/neuralNetwork/testnn_neuralNetwork_verify.m exactly.
    Uses two ACASXU instances from vnn-comp 2023 for testing.
    """
    # MATLAB: resSafe = example_neuralNetwork_verify_safe(); % Verify the specifiation.
    # MATLAB: assert(strcmp(resSafe,'VERIFIED'));
    resSafe = example_neuralNetwork_verify_safe()
    assert resSafe == 'VERIFIED', f"Expected 'VERIFIED', got '{resSafe}'"
    
    # MATLAB: resUnsafe = example_neuralNetwork_verify_unsafe(); % Find a counterexample.
    # MATLAB: assert(strcmp(resUnsafe,'COUNTEREXAMPLE'));
    resUnsafe = example_neuralNetwork_verify_unsafe()
    assert resUnsafe == 'COUNTEREXAMPLE', f"Expected 'COUNTEREXAMPLE', got '{resUnsafe}'"


def example_neuralNetwork_verify_safe():
    """
    Example for the verification of a neural networks using the function neuralNetwork/verify.
    (MATLAB example_neuralNetwork_verify_safe equivalent)
    
    Returns:
        str: Verification result ['VERIFIED','COUNTEREXAMPLE','UNKNOWN']
    """
    import time
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    from cora_python.g.macros.CORAROOT import CORAROOT
    
    # MATLAB: rng('default')
    np.random.seed(0)  # MATLAB's 'default' seed
    
    verbose = True
    # MATLAB: modelPath = 'ACASXU_run2a_1_2_batch_2000.onnx';
    # MATLAB: specPath = 'prop_1.vnnlib';
    cora_root = CORAROOT()
    modelPath = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx')
    specPath = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'prop_1.vnnlib')
    timeout = 2  # MATLAB: timeout = 2;
    
    # Check if files exist
    if not os.path.isfile(modelPath) or not os.path.isfile(specPath):
        pytest.skip(f"Required files not found: modelPath={modelPath}, specPath={specPath}")
    
    # MATLAB: [nn,x,r,A,b,safeSet,options] = aux_readModelAndSpecs(modelPath,specPath);
    nn, x, r, A, b, safeSet, options = aux_readModelAndSpecs(modelPath, specPath)
    
    # MATLAB: timerVal = tic;
    # MATLAB: [res,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);
    timerVal = time.time()
    res, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    
    # MATLAB: if verbose ... fprintf('%s -- %s: %s\n',modelPath,specPath,res); ...
    if verbose:
        elapsed_time = time.time() - timerVal
        print(f'{os.path.basename(modelPath)} -- {os.path.basename(specPath)}: {res}')
        print(f'--- Verification time: {elapsed_time:.4f} / {timeout:.4f} [s]')
    
    # MATLAB: aux_writeResults(res,x_,y_);
    aux_writeResults(res, x_, y_)
    
    return res


def example_neuralNetwork_verify_unsafe():
    """
    Example for finding a counter example using the function neuralNetwork/verify.
    (MATLAB example_neuralNetwork_verify_unsafe equivalent)
    
    Returns:
        str: Verification result ['VERIFIED','COUNTEREXAMPLE','UNKNOWN']
    """
    import time
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    from cora_python.g.macros.CORAROOT import CORAROOT
    
    # MATLAB: rng('default')
    np.random.seed(0)  # MATLAB's 'default' seed
    
    verbose = True
    # MATLAB: modelPath = 'ACASXU_run2a_1_2_batch_2000.onnx';
    # MATLAB: specPath = 'prop_2.vnnlib';
    cora_root = CORAROOT()
    modelPath = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx')
    specPath = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'prop_2.vnnlib')
    timeout = 2  # MATLAB: timeout = 2; (but uses 10 in actual verify call)
    
    # Check if files exist
    if not os.path.isfile(modelPath) or not os.path.isfile(specPath):
        pytest.skip(f"Required files not found: modelPath={modelPath}, specPath={specPath}")
    
    # MATLAB: [nn,x,r,A,b,safeSet,options] = aux_readModelAndSpecs(modelPath,specPath);
    nn, x, r, A, b, safeSet, options = aux_readModelAndSpecs(modelPath, specPath)
    
    # MATLAB: timerVal = tic;
    # MATLAB: [res,x_,y_] = nn.verify(x,r,A,b,safeSet,options,10,verbose);
    timerVal = time.time()
    res, x_, y_ = nn.verify(x, r, A, b, safeSet, options, 10, verbose)  # MATLAB uses 10, not timeout
    
    # MATLAB: if verbose ... fprintf('%s -- %s: %s\n',modelPath,specPath,res); ...
    if verbose:
        elapsed_time = time.time() - timerVal
        print(f'{os.path.basename(modelPath)} -- {os.path.basename(specPath)}: {res}')
        print(f'--- Verification time: {elapsed_time:.4f} / {timeout:.4f} [s]')
    
    # MATLAB: aux_writeResults(res,x_,y_);
    aux_writeResults(res, x_, y_)
    
    return res


def aux_readModelAndSpecs(modelPath: str, specPath: str):
    """
    Read model and specs (MATLAB aux_readModelAndSpecs equivalent)
    
    MATLAB signature:
    function [nn,x,r,A,b,safeSet,options] = aux_readModelAndSpecs(modelPath,specPath)
    """
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    
    # MATLAB: nn = neuralNetwork.readONNXNetwork(modelPath,false,'BSSC');
    nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BSSC')
    
    # MATLAB: [X0,specs] = vnnlib2cora(specPath);
    X0, specs = vnnlib2cora(specPath)
    
    # MATLAB: x = 1/2*(X0{1}.sup + X0{1}.inf);
    # MATLAB: r = 1/2*(X0{1}.sup - X0{1}.inf);
    x = 0.5 * (X0[0].sup + X0[0].inf)
    r = 0.5 * (X0[0].sup - X0[0].inf)
    
    # MATLAB: if isa(specs.set,'halfspace')
    #     A = specs.set.c';
    #     b = -specs.set.d;
    # else
    #     A = specs.set.A;
    #     b = -specs.set.b;
    # end
    from cora_python.contSet.polytope.representsa_ import representsa_
    isHalfspace = representsa_(specs.set, 'halfspace')
    
    if isHalfspace:
        A = specs.set.A  # (1, n) - this is c' in MATLAB
        b = -specs.set.b.flatten()  # (1,) - this is d in MATLAB, NEGATED to match MATLAB
        if b.ndim == 0:
            b = np.array([b])
    else:
        A = specs.set.A
        b = -specs.set.b  # NEGATED to match MATLAB
        if b.ndim == 1:
            b = b.reshape(-1, 1)
    
    # MATLAB: safeSet = strcmp(specs.type,'safeSet');
    safeSet = (specs.type == 'safeSet')
    
    # MATLAB: options.nn = struct('use_approx_error',true,'poly_method','bounds','train',struct('backprop',false,'mini_batch_size',512));
    options = {}
    options['nn'] = {
        'use_approx_error': True,
        'poly_method': 'bounds',
        'train': {
            'backprop': False,
            'mini_batch_size': 512
        }
    }
    # MATLAB: options = nnHelper.validateNNoptions(options,true);
    options = validateNNoptions(options, True)
    # MATLAB: options.nn.interval_center = false;
    options['nn']['interval_center'] = False
    
    return nn, x, r, A, b, safeSet, options


def aux_writeResults(res: str, x_, y_):
    """
    Write results (MATLAB aux_writeResults equivalent)
    
    MATLAB signature:
    function aux_writeResults(res,x_,y_)
    """
    # MATLAB: if strcmp(res,'VERIFIED')
    #     fprintf(['unsat' newline]);
    if res == 'VERIFIED':
        print('unsat')
    # MATLAB: elseif strcmp(res,'COUNTEREXAMPLE')
    #     fprintf(['sat' newline '(']);
    #     for j=1:size(x_,1)
    #         fprintf(['(X_%d %f)' newline],j-1,x_(j));
    #     end
    #     for j=1:size(y_,1)
    #         fprintf(['(Y_%d %f)' newline],j-1,y_(j));
    #     end
    #     fprintf(')');
    elif res == 'COUNTEREXAMPLE':
        print('sat')
        print('(')
        if x_ is not None and x_.size > 0:
            for j in range(x_.shape[0]):
                print(f'(X_{j} {x_[j, 0]:f})')
        if y_ is not None and y_.size > 0:
            for j in range(y_.shape[0]):
                print(f'(Y_{j} {y_[j, 0]:f})')
        print(')')
    # MATLAB: else
    #     fprintf(['unknown' newline]);
    else:
        print('unknown')


def testnn_verify():
    """
    Test neuralNetwork.verify function (MATLAB testnn_verify equivalent)
    
    This test matches cora_matlab/unitTests/nn/testnn_verify.m exactly.
    Uses specs from the ACASXU benchmark: prop_1, prop_2.
    """
    # We use the specs from the acasxu benchmark: prop_1, prop_2, prop_3, and prop_5.
    
    cora_root = CORAROOT()
    modelPath = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx')
    timeout = 100  # MATLAB: timeout = 100;
    
    # Check if files exist, skip test if they don't
    if not os.path.isfile(modelPath):
        pytest.skip(f"ACASXU model file not found: {modelPath}")
    
    # First test case: prop_1.vnnlib
    prop1Filename = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'prop_1.vnnlib')
    if not os.path.isfile(prop1Filename):
        pytest.skip(f"VNNLIB file not found: {prop1Filename}")
    
    nn, options, x, r, A, b, safeSet = aux_readNetworkAndOptions(modelPath, prop1Filename)
    # Do verification.
    # MATLAB: [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,false);
    verifRes, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, False)
    # MATLAB: assert(strcmp(verifRes,'VERIFIED') & isempty(x_) & isempty(y_));
    assert verifRes == 'VERIFIED', f"Expected 'VERIFIED', got '{verifRes}'"
    assert x_ is None or (hasattr(x_, 'size') and x_.size == 0), f"Expected empty x_, got {x_}"
    assert y_ is None or (hasattr(y_, 'size') and y_.size == 0), f"Expected empty y_, got {y_}"
    
    # Second test case: prop_2.vnnlib
    prop2Filename = os.path.join(cora_root, 'cora_matlab', 'models', 'Cora', 'nn', 'prop_2.vnnlib')
    if not os.path.isfile(prop2Filename):
        pytest.skip(f"VNNLIB file not found: {prop2Filename}")
    
    nn, options, x, r, A, b, safeSet = aux_readNetworkAndOptions(modelPath, prop2Filename)
    # Do verification.
    # MATLAB: [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,false);
    verifRes, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, False)
    # MATLAB: assert(strcmp(verifRes,'COUNTEREXAMPLE') & ~isempty(x_) & ~isempty(y_));
    assert verifRes == 'COUNTEREXAMPLE', f"Expected 'COUNTEREXAMPLE', got '{verifRes}'"
    assert x_ is not None and x_.size > 0, f"Expected non-empty x_, got {x_}"
    assert y_ is not None and y_.size > 0, f"Expected non-empty y_, got {y_}"
    # MATLAB: assert(aux_checkCounterexample(nn,A,b,safeSet,x_,y_));
    assert aux_checkCounterexample(nn, A, b, safeSet, x_, y_)


# Auxiliary functions -----------------------------------------------------

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
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    
    # Create evaluation options.
    # MATLAB: options.nn = struct('use_approx_error',true,'poly_method','bounds','train',struct('backprop',false,'mini_batch_size',512));
    options = {}
    options['nn'] = {
        'use_approx_error': True,
        'poly_method': 'bounds',
        'train': {
            'backprop': False,
            'mini_batch_size': 512  # MATLAB: 512
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
    #     b = -specs.set.d;
    # else
    #     A = specs.set.A;
    #     b = -specs.set.b;
    # end
    # In Python, halfspace is deprecated and represented as a polytope
    # Check if polytope represents a halfspace using representsa_
    from cora_python.contSet.polytope.representsa_ import representsa_
    isHalfspace = representsa_(specs.set, 'halfspace')
    
    if isHalfspace:
        # Halfspace case: c' * x <= d
        # MATLAB: A = specs.set.c';
        # MATLAB: b = -specs.set.d;  (NOTE: MATLAB negates d!)
        # In Python, a halfspace polytope has A (1, n) and b (1, 1)
        # Extract as c' and d to match MATLAB
        A = specs.set.A  # (1, n) - this is c' in MATLAB
        b = -specs.set.b.flatten()  # (1,) - this is d in MATLAB, NEGATED to match MATLAB
        if b.ndim == 0:
            b = np.array([b])
    else:
        # Polytope case: A * x <= b
        # MATLAB: A = specs.set.A;
        # MATLAB: b = -specs.set.b;  (NOTE: MATLAB negates b!)
        A = specs.set.A
        b = -specs.set.b  # NEGATED to match MATLAB
        if b.ndim == 1:
            b = b.reshape(-1, 1)
    
    # MATLAB: safeSet = strcmp(specs.type,'safeSet');
    safeSet = (specs.type == 'safeSet')
    
    return nn, options, x, r, A, b, safeSet


def aux_checkCounterexample(nn, A, b, safeSet, x_, y_):
    """
    Check if counterexample is valid (MATLAB aux_checkCounterexample equivalent)
    
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
    #     violates = any(A*yi + b >= 0,1);
    # else
    #     violates = all(A*yi + b <= 0,1);
    # end
    # Ensure shapes are correct for matrix multiplication
    # A: (num_constraints, num_outputs)
    # yi: (num_outputs, 1) or (num_outputs,)
    # b: (num_constraints, 1) or (num_constraints,)
    if yi.ndim == 1:
        yi = yi.reshape(-1, 1)  # (num_outputs,) -> (num_outputs, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)  # (num_constraints,) -> (num_constraints, 1)
    
    # Compute A*yi + b
    # A @ yi: (num_constraints, num_outputs) @ (num_outputs, 1) = (num_constraints, 1)
    # A @ yi + b: (num_constraints, 1) + (num_constraints, 1) = (num_constraints, 1)
    ld = A @ yi + b  # (num_constraints, 1)
    
    # Debug output
    print(f"\naux_checkCounterexample DEBUG:")
    print(f"  A shape: {A.shape}, yi shape: {yi.shape}, b shape: {b.shape}")
    print(f"  ld shape: {ld.shape}, ld values: {ld.flatten()}")
    print(f"  safeSet: {safeSet}")
    print(f"  y_ shape: {y_.shape}, yi shape: {yi.shape}")
    print(f"  y_ values: {y_.flatten()}")
    print(f"  yi values: {yi.flatten()}")
    print(f"  abs(y_ - yi): {np.abs(y_ - yi).flatten()}")
    
    if safeSet:
        # For safe set: violation means any(A*yi + b >= 0)
        # MATLAB: violates = any(A*yi + b >= 0,1);
        # MATLAB checks along dimension 1 (columns), which is axis=1 in Python
        # But since ld is (num_constraints, 1), we check all elements
        violates = np.any(ld >= 0)
        print(f"  For safeSet: any(ld >= 0) = {violates}")
    else:
        # For unsafe set: violation means all(A*yi + b <= 0)
        # MATLAB: violates = all(A*yi + b <= 0,1);
        # MATLAB checks along dimension 1 (columns), which is axis=1 in Python
        # But since ld is (num_constraints, 1), we check all elements
        violates = np.all(ld <= 0)
        print(f"  For unsafeSet: all(ld <= 0) = {violates}, ld <= 0: {(ld <= 0).flatten()}")
    
    # MATLAB: assert(res & violates);
    assert res and violates, f"Counterexample check failed: res={res}, violates={violates}"
    return True
