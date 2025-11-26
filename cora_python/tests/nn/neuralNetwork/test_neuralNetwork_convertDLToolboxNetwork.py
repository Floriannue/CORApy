"""
test_neuralNetwork_convertDLToolboxNetwork - tests the conversion 
   to and from networks from the Matlab DL toolbox

Description:
    This test verifies that ONNX networks can be correctly loaded and converted
    to CORA neural networks. Since Python doesn't have MATLAB's Deep Learning
    Toolbox, we compare CORA network evaluation against ONNX Runtime evaluation
    to verify correctness.

Syntax:
    pytest test_neuralNetwork_convertDLToolboxNetwork.py

Inputs:
    -

Outputs:
    res - boolean (via pytest assertions)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       15-May-2025
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import pytest
import numpy as np
import os
import onnxruntime as ort

from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.nn.neuralNetwork import NeuralNetwork


def _evaluate_onnx_runtime(model_path, x, input_shape=None):
    """
    Evaluate ONNX model using ONNX Runtime (equivalent to MATLAB's DLToolbox predict)
    
    This matches MATLAB's behavior:
    - Feed-forward: nn_dlt.predict(x')' -> transpose input, transpose output
    - Convolutional: nn_dlt.predict(reshape(x, inputSize))' -> reshape, transpose output
    
    Args:
        model_path: Path to ONNX model file
        x: Input vector (column vector, neurons_in x 1) - matches MATLAB's x
        input_shape: Optional input shape for reshaping (for convolutional networks)
                    - matches MATLAB's nn.layers{1}.inputSize
                    - CORA format: [H, W, C] for BCSS/BSSC, [features] for BC
        
    Returns:
        Output vector (column vector, neurons_out x 1) - matches MATLAB's result after transpose
    """
    
    # Load ONNX model
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    onnx_input_shape = sess.get_inputs()[0].shape
    
    # Prepare input (matches MATLAB exactly)
    if input_shape is not None:
        # Convolutional: reshape(x, nn.layers{1}.inputSize) - matches MATLAB line 51, 75
        # CORA inputSize is [H, W, C] for BCSS/BSSC, or [features] for feed-forward
        x_flat = x.flatten() if x.ndim > 1 else x
        
        if len(input_shape) == 3:
            # 3D input shape: [H, W, C] - convolutional network
            H, W, C = input_shape
            x_reshaped = x_flat.reshape(H, W, C)  # [H, W, C]
            
            # Determine format from ONNX input shape
            # BCSS format: ONNX expects [B, C, H, W]
            # BSSC format: ONNX expects [B, H, W, C]
            if len(onnx_input_shape) == 4:
                # Check which format by comparing dimensions
                # For BCSS: dim[1] = C, dim[2] = H, dim[3] = W
                # For BSSC: dim[1] = H, dim[2] = W, dim[3] = C
                # Handle dynamic dimensions (can be int, None, or string)
                onnx_dims = []
                for dim in onnx_input_shape:
                    if isinstance(dim, int) and dim > 0:
                        onnx_dims.append(dim)
                    else:
                        onnx_dims.append(None)
                
                # Try BCSS first: check if dim[1] matches C and dim[2] matches H
                if onnx_dims[1] == C and (onnx_dims[2] == H or onnx_dims[2] is None):
                    # BCSS format: [B, C, H, W]
                    x_reshaped = x_reshaped.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
                    x_reshaped = np.expand_dims(x_reshaped, axis=0)  # [C, H, W] -> [1, C, H, W]
                elif onnx_dims[1] == H and (onnx_dims[2] == W or onnx_dims[2] is None):
                    # BSSC format: [B, H, W, C]
                    x_reshaped = np.expand_dims(x_reshaped, axis=0)  # [H, W, C] -> [1, H, W, C]
                else:
                    # Default to BCSS if shape doesn't match (most common format)
                    x_reshaped = x_reshaped.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
                    x_reshaped = np.expand_dims(x_reshaped, axis=0)  # [C, H, W] -> [1, C, H, W]
            else:
                # Default to BCSS if shape doesn't match
                x_reshaped = x_reshaped.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
                x_reshaped = np.expand_dims(x_reshaped, axis=0)  # [C, H, W] -> [1, C, H, W]
        elif len(input_shape) == 1:
            # 1D input shape: [features] - feed-forward network
            # ONNX expects [batch, features] for BC format
            # But if ONNX expects 4D, might need to reshape differently
            if len(onnx_input_shape) == 2:
                # Standard feed-forward: [batch, features]
                x_reshaped = x_flat.reshape(1, -1)  # [1, features]
            elif len(onnx_input_shape) == 4:
                # ONNX expects 4D, need to reshape to match
                # Try to infer from ONNX shape (handle dynamic dimensions)
                onnx_dims = []
                for dim in onnx_input_shape:
                    if isinstance(dim, int) and dim > 0:
                        onnx_dims.append(dim)
                    else:
                        onnx_dims.append(1)  # Use 1 for dynamic dimensions
                
                total_expected = np.prod(onnx_dims[1:])  # Exclude batch dimension
                if total_expected == x_flat.size:
                    # Reshape to match ONNX expected shape (excluding batch)
                    remaining_dims = tuple(onnx_dims[1:])
                    x_reshaped = x_flat.reshape(*remaining_dims)
                    x_reshaped = np.expand_dims(x_reshaped, axis=0)  # Add batch
                else:
                    # Fallback: just use [1, features]
                    x_reshaped = x_flat.reshape(1, -1)
            else:
                # Default: [batch, features]
                x_reshaped = x_flat.reshape(1, -1)
        else:
            # Other shapes: try to match ONNX expected shape
            x_reshaped = x_flat.reshape(*input_shape)
            # Add batch dimension if needed
            if len(onnx_input_shape) > len(x_reshaped.shape):
                x_reshaped = np.expand_dims(x_reshaped, axis=0)
    else:
        # Feed-forward: x' (transpose) - matches MATLAB line 39
        # x is (neurons_in, 1), transpose to (1, neurons_in) for ONNX [batch, features]
        # But check if ONNX expects 4D
        if len(onnx_input_shape) == 4:
            # ONNX expects 4D, try to reshape
            onnx_dims = []
            for dim in onnx_input_shape:
                if isinstance(dim, int) and dim > 0:
                    onnx_dims.append(dim)
                else:
                    onnx_dims.append(1)  # Use 1 for dynamic dimensions
            
            total_expected = np.prod(onnx_dims[1:])  # Exclude batch dimension
            if total_expected == x.size:
                remaining_dims = tuple(onnx_dims[1:])
                x_reshaped = x.flatten().reshape(*remaining_dims)
                x_reshaped = np.expand_dims(x_reshaped, axis=0)
            else:
                x_reshaped = x.T
        else:
            x_reshaped = x.T
    
    # Ensure float32 (ONNX standard)
    x_reshaped = x_reshaped.astype(np.float32)
    
    # Run inference (equivalent to MATLAB's predict)
    outputs = sess.run(None, {input_name: x_reshaped})
    y_onnx = outputs[0]
    
    # Transpose output to column vector (matches MATLAB: ...)'
    # ONNX output is [batch, features], we need [features, 1]
    if y_onnx.ndim == 1:
        y_onnx = y_onnx.reshape(-1, 1)
    else:
        # Take first batch and convert to column vector
        y_onnx = y_onnx[0].reshape(-1, 1)
    
    return y_onnx


def test_feedforward_neural_network():
    """
    Test conversion of feed-forward ONNX network to CORA network
    
    This test matches the MATLAB test structure for feed-forward networks.
    It compares CORA evaluation against ONNX Runtime evaluation.
    """
    # high tol due to DLT using singles (matches MATLAB test)
    tol = 1e-6
    
    # load network (matches MATLAB: neuralNetwork.readONNXNetwork('nn-nav-set.onnx'))
    model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'nn-nav-set.onnx')
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")
    
    nn = NeuralNetwork.readONNXNetwork(model_path)
    
    # test network (matches MATLAB test)
    x = np.ones((nn.neurons_in, 1))
    y = nn.evaluate(x)
    y_onnx = _evaluate_onnx_runtime(model_path, x)
    
    # Compare results (matches MATLAB: assert(all(withinTol(y,y_dlt,tol))))
    assert np.all(withinTol(y, y_onnx, tol)), f"CORA and ONNX Runtime outputs differ: CORA={y.flatten()}, ONNX={y_onnx.flatten()}"


def test_convolutional_neural_network():
    """
    Test conversion of convolutional ONNX network to CORA network
    
    This test matches the MATLAB test structure for convolutional networks.
    It compares CORA evaluation against ONNX Runtime evaluation.
    """
    # high tol due to DLT using singles (matches MATLAB test)
    tol = 1e-6
    
    # load network (matches MATLAB: neuralNetwork.readONNXNetwork('vnn_verivital_avgpool.onnx',false,'BCSS'))
    model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")
    
    nn = NeuralNetwork.readONNXNetwork(model_path, False, 'BCSS')
    
    # test network (matches MATLAB test)
    x = np.ones((nn.neurons_in, 1))
    y = nn.evaluate(x)
    
    # Get input shape from first layer (matches MATLAB: reshape(x,nn.layers{1}.inputSize))
    input_shape = None
    if hasattr(nn.layers[0], 'inputSize') and nn.layers[0].inputSize:
        input_shape = nn.layers[0].inputSize
    
    y_onnx = _evaluate_onnx_runtime(model_path, x, input_shape)
    
    # Compare results (matches MATLAB: assert(all(withinTol(y,y_dlt,tol))))
    assert np.all(withinTol(y, y_onnx, tol)), f"CORA and ONNX Runtime outputs differ: CORA={y.flatten()}, ONNX={y_onnx.flatten()}"


def test_vnncomp_neural_networks():
    """
    Test conversion of VNN-COMP ONNX networks to CORA networks
    
    This test matches the MATLAB test structure for VNN-COMP networks.
    It compares CORA evaluation against ONNX Runtime evaluation for multiple networks.
    """
    # high tol due to DLT using singles (matches MATLAB test)
    tol = 1e-6
    
    # Reset the random number generator (matches MATLAB: rng('default'))
    np.random.seed(0)
    
    # Specify the model paths (matches MATLAB test exactly)
    model_paths = [
        os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx'),
        os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'ACASXU_run2a_5_3_batch_2000.onnx'),
    ]
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            pytest.skip(f"Model file not found: {model_path}")
        
        # Load the neural network as a CORA network (matches MATLAB: neuralNetwork.readONNXNetwork(modelpaths{i},false,'BSSC'))
        nn = NeuralNetwork.readONNXNetwork(model_path, False, 'BSSC')
        
        # Generate a random input (matches MATLAB: rand(nn.neurons_in,1))
        x = np.random.rand(nn.neurons_in, 1)
        
        # Compute the output with the CORA network (matches MATLAB: y = nn.evaluate(x))
        y = nn.evaluate(x)
        
        # Get input shape from first layer (matches MATLAB: reshape(x,nn.layers{1}.inputSize))
        input_shape = None
        if hasattr(nn.layers[0], 'inputSize') and nn.layers[0].inputSize:
            input_shape = nn.layers[0].inputSize
        
        # Compute the output with ONNX Runtime (matches MATLAB: nn_dlt.predict(...))
        y_onnx = _evaluate_onnx_runtime(model_path, x, input_shape)
        
        # Check if the results are within the tolerance (matches MATLAB: assert(all(withinTol(y,y_dlt,tol))))
        assert np.all(withinTol(y, y_onnx, tol)), f"CORA and ONNX Runtime outputs differ for {os.path.basename(model_path)}: CORA={y.flatten()}, ONNX={y_onnx.flatten()}"


if __name__ == "__main__":
    # Run all tests (matches MATLAB test structure)
    test_feedforward_neural_network()
    print("test_feedforward_neural_network successful")
    
    test_convolutional_neural_network()
    print("test_convolutional_neural_network successful")
    
    test_vnncomp_neural_networks()
    print("test_vnncomp_neural_networks successful")
    
    print("All tests successful")

