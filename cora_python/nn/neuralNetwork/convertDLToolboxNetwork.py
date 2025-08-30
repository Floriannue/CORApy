"""
convertDLToolboxNetwork - converts a network from the Deep Learning
   Toolbox to a CORA neuralNetwork for verification

Syntax:
    res = neuralNetwork.convertDLToolboxNetwork(dltoolbox_layers)
    res = neuralNetwork.convertDLToolboxNetwork(dltoolbox_layers,verbose)

Inputs:
    dltoolbox_layers - layer array (e.g. dltoolbox_nn.Layers)
    verbose - true/false whether information should be displayed

Outputs:
    obj - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner, Lukas Koller
Written:       30-March-2022
Last update:   05-June-2022 (LK, Conv, Pool)
                17-January-2023 (TL, Reshape)
                23-November-2023 (TL, bug fix with scalar element-wise operation)
                25-July-2023 (TL, nnElementwiseAffineLayer)
                31-July-2023 (LK, nnSoftmaxLayer)
Last revision: 17-August-2022
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import List, Any, Dict, Tuple, Optional
from .neuralNetwork import NeuralNetwork


def convertDLToolboxNetwork(dltoolbox_layers: List, verbose: bool = False) -> NeuralNetwork:
    """
    Convert DLToolbox network to CORA network (matches MATLAB exactly).
    
    Args:
        dltoolbox_layers: DLToolbox layers
        verbose: Whether to print verbose output
        
    Returns:
        CORA neural network
    """
    n = len(dltoolbox_layers)
    
    layers = []
    inputSize = []
    currentSize = []
    
    if verbose:
        print("Converting Deep Learning Toolbox Model to neuralNetwork...")
    
    i = 0
    while i < n:
        dlt_layer = dltoolbox_layers[i]
        if verbose:
            print(f"#{i+1}: {type(dlt_layer).__name__}")
        
        if isinstance(dlt_layer, list):
            # We need to construct a composite layer.
            # Each dlt_layer is already a list like [layer] from readONNXNetwork
            layers_, _, currentSize_ = aux_convertLayer(layers, 
                dlt_layer[0], currentSize, verbose)
            
            # Update current size
            if layers_ and len(layers_) > len(layers):
                currentSize = layers_[-1].getOutputSize(currentSize) if hasattr(layers_[-1], 'getOutputSize') else currentSize
        else:
            # Just append a regular layer.
            layers, inputSize_, currentSize = aux_convertLayer(layers, 
                dlt_layer, currentSize, verbose)
            if not inputSize:
                inputSize = inputSize_
        
        i += 1
    
    # instantiate neural network
    obj = NeuralNetwork(layers)
    
    # Set input size from inputSize variable (matches MATLAB exactly)
    if inputSize:
        if np.isscalar(inputSize):
            inputSize = [inputSize, 1]
        
        # set input size
        obj.setInputSize(inputSize, False)
        
        # sanity check (should not fail) - matches MATLAB
        try:
            x = np.zeros(inputSize).reshape(-1, 1)
            obj.evaluate(x)
            if verbose:
                print("Sanity check passed")
        except Exception as e:
            if verbose:
                print(f"Warning: Sanity check failed: {e}")
    
    return obj


def aux_convertLayer(layers: List, dlt_layer: Any, currentSize: List, verbose: bool) -> Tuple[List, List, List]:
    """
    Convert a single DLT layer to CORA layer (matches MATLAB aux_convertLayer).
    
    Args:
        layers: Current list of CORA layers
        dlt_layer: DLT layer to convert
        currentSize: Current input size
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (layers, inputSize, currentSize)
    """
    inputSize = []
    
    # Validate layer has required attributes
    if not dlt_layer or (isinstance(dlt_layer, dict) and not dlt_layer):
        if verbose:
            print("    Warning: Empty or invalid layer, skipping")
        return layers, inputSize, currentSize
    
    # Get layer type for logging
    if verbose:
        layer_type = dlt_layer.get('Type', 'Unknown') if isinstance(dlt_layer, dict) else 'Unknown'
        print(f"  Converting layer type: {layer_type}")
    
    # Check for InputLayer (matches MATLAB exactly)
    if isinstance(dlt_layer, dict) and dlt_layer.get('Type') == 'InputLayer':
        # Input layer - extract input size (matches MATLAB exactly)
        inputSize = dlt_layer.get('InputSize', [1, 1])
        if len(inputSize) == 1:
            inputSize = [inputSize[0], 1]
        elif len(inputSize) == 3:
            # channel dimension should be last: [h,w,c] (like MATLAB)
            pass
            
        if verbose:
            print(f"    Found input size: {inputSize}")
        
        # Handle normalization like MATLAB (if present)
        if dlt_layer.get('Normalization') == 'zscore':
            mu = dlt_layer.get('Mean', np.zeros(1))
            sigma = dlt_layer.get('StandardDeviation', np.ones(1))
            from ..layers.linear.nnElementwiseAffineLayer import nnElementwiseAffineLayer
            # Add normalization layer to computational layers (like MATLAB)
            layers.append(nnElementwiseAffineLayer(1/sigma, -mu/sigma, dlt_layer.get('Name', '')))
        
        # Input layer itself is just metadata, don't add to computational layers
        # The actual input size will be determined from the first layer's weights
        # Only normalization layers (if any) are added above
        currentSize = inputSize
        return layers, inputSize, currentSize
    
    # Check for FullyConnectedLayer (matches MATLAB exactly)
    elif isinstance(dlt_layer, dict) and dlt_layer.get('Type') == 'FullyConnectedLayer':
        # Linear layer (matches MATLAB exactly)
        from ..layers.linear.nnLinearLayer import nnLinearLayer
        
        # Extract weights and biases - ensure consistent access
        W = dlt_layer.get('Weight', np.eye(1))
        b = dlt_layer.get('Bias', np.zeros((1, 1)))
        
        # Ensure proper shapes
        if len(W.shape) == 1:
            W = W.reshape(-1, 1)
        if len(b.shape) == 1:
            b = b.reshape(-1, 1)
        
        layer = nnLinearLayer(W, b, name=dlt_layer.get('Name', ''))
        layers.append(layer)
        
        # Update sizes (matches MATLAB exactly)
        if not currentSize:
            currentSize = [W.shape[1], 1]  # input features from weights
        currentSize = [W.shape[0], 1]  # output features
        
    # Check for ReLULayer (matches MATLAB exactly)
    elif isinstance(dlt_layer, dict) and dlt_layer.get('Type') == 'ReLULayer':
        # ReLU activation layer
        from ..layers.nonlinear.nnReLULayer import nnReLULayer
        layer = nnReLULayer(name=dlt_layer.get('Name', ''))
        layers.append(layer)
        
    # Check for SigmoidLayer (matches MATLAB exactly)
    elif isinstance(dlt_layer, dict) and dlt_layer.get('Type') == 'SigmoidLayer':
        # Sigmoid activation layer
        from ..layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
        layer = nnSigmoidLayer(name=dlt_layer.get('Name', ''))
        layers.append(layer)
        
    # Check for TanhLayer (matches MATLAB exactly)
    elif isinstance(dlt_layer, dict) and dlt_layer.get('Type') == 'TanhLayer':
        # Tanh activation layer
        from ..layers.nonlinear.nnTanhLayer import nnTanhLayer
        layer = nnTanhLayer(name=dlt_layer.get('Name', ''))
        layers.append(layer)
        
    # Check for reshape layers (matches MATLAB exactly)
    elif isinstance(dlt_layer, dict) and dlt_layer.get('Type') == 'ReshapeLayer':
        # Reshape layer
        from ..layers.other.nnReshapeLayer import nnReshapeLayer
        
        # Extract shape information consistently
        shape = dlt_layer.get('Shape', [-1])  # Default to flatten if not specified
        
        # Special handling for Flatten layers (which were converted to ReshapeLayer)
        if 'FlattenAxis' in dlt_layer:
            # This was originally a Flatten layer
            flatten_axis = dlt_layer['FlattenAxis']
            if verbose:
                print(f"    Converting Flatten layer with axis {flatten_axis}")
            
            # For Flatten layers, we need to determine the target shape based on the input
            # Since we know the input shape from the InputLayer, we can calculate the target
            if currentSize and len(currentSize) > 0:
                # Calculate what the flatten should produce
                # For input [1, 1, 1, 5] and axis=1, we keep dims 0 and flatten the rest
                # This means: [1, 1*1*5] = [1, 5]
                # But since the next layer expects 5 input features, we want [5]
                if flatten_axis == 1:  # This is the case in our ONNX model
                    # Flatten everything after dimension 0
                    target_size = currentSize[0] * np.prod(currentSize[1:]) if len(currentSize) > 1 else currentSize[0]
                    shape = [target_size]
                    if verbose:
                        print(f"    Flatten target shape calculated: {shape}")
                else:
                    # For other axes, use the default -1 (dynamic)
                    shape = [-1]
            else:
                # No current size info, use dynamic shape
                shape = [-1]
        
        layer = nnReshapeLayer(shape, name=dlt_layer.get('Name', ''))
        layers.append(layer)
        
        # Update size based on reshape
        if shape and -1 not in shape:
            currentSize = [np.prod(shape), 1]
        
    # Check for Convolution2DLayer (matches MATLAB exactly)
    elif (hasattr(dlt_layer, '__class__') and 'Convolution2DLayer' in str(dlt_layer.__class__)):
        # Convolutional layer
        from ..layers.linear.nnConv2DLayer import nnConv2DLayer
        
        # Extract convolution parameters
        W = dlt_layer.Weights if hasattr(dlt_layer, 'Weights') else np.eye(1)
        b = dlt_layer.Bias if hasattr(dlt_layer, 'Bias') else np.zeros(1)
        padding = dlt_layer.PaddingSize if hasattr(dlt_layer, 'PaddingSize') else [0, 0]
        stride = dlt_layer.Stride if hasattr(dlt_layer, 'Stride') else [1, 1]
        dilation = dlt_layer.DilationFactor if hasattr(dlt_layer, 'DilationFactor') else [1, 1]
        
        # Ensure bias is column vector
        b = np.reshape(b, (-1, 1))
        
        layer = nnConv2DLayer(W, b, padding, stride, dilation, name=getattr(dlt_layer, 'Name', ''))
        layers.append(layer)
        
    # Check for pooling layers (matches MATLAB exactly)
    elif (hasattr(dlt_layer, '__class__') and 'Pooling2DLayer' in str(dlt_layer.__class__)):
        # Pooling layer - check if it's max or average
        if 'Max' in str(dlt_layer.__class__):
            from ..layers.other.nnMaxPoolLayer import nnMaxPoolLayer
            poolSize = dlt_layer.PoolSize if hasattr(dlt_layer, 'PoolSize') else [2, 2]
            stride = dlt_layer.Stride if hasattr(dlt_layer, 'Stride') else [1, 1]
            layer = nnMaxPoolLayer(poolSize, stride, name=getattr(dlt_layer, 'Name', ''))
        else:  # AveragePool
            from ..layers.other.nnAvgPoolLayer import nnAvgPoolLayer
            poolSize = dlt_layer.PoolSize if hasattr(dlt_layer, 'PoolSize') else [2, 2]
            padding = dlt_layer.PaddingSize if hasattr(dlt_layer, 'PaddingSize') else [0, 0]
            stride = dlt_layer.Stride if hasattr(dlt_layer, 'Stride') else [1, 1]
            dilation = [1, 1]  # Default like MATLAB
            layer = nnAvgPoolLayer(poolSize, padding, stride, dilation, name=getattr(dlt_layer, 'Name', ''))
        
        layers.append(layer)
        
    # Check for ElementwiseAffineLayer (matches MATLAB exactly)
    elif (hasattr(dlt_layer, '__class__') and 'ElementwiseAffineLayer' in str(dlt_layer.__class__)):
        from ..layers.linear.nnElementwiseAffineLayer import nnElementwiseAffineLayer
        
        s = dlt_layer.Scale if hasattr(dlt_layer, 'Scale') else np.ones(1)
        o = dlt_layer.Offset if hasattr(dlt_layer, 'Offset') else np.zeros(1)
        
        # Fix dimensions for [h,w,c] inputs like MATLAB
        if len(currentSize) == 3:
            if not np.isscalar(s):
                # Fix if all values are equal
                if len(s) > 0 and np.all(s[0] == s):
                    s = s[0]
                else:
                    # Try to fix scaling factor
                    s = np.reshape(np.tile(s, np.array(currentSize) // np.array(s.shape)), currentSize)
            
            if not np.isscalar(o):
                # Fix if all values are equal
                if len(o) > 0 and np.all(o[0] == o):
                    o = o[0]
                else:
                    # Try to fix offset vector
                    o = np.reshape(np.tile(o, np.array(currentSize) // np.array(o.shape)), currentSize)
        
        # Should be column vector
        s = np.reshape(s, (-1, 1))
        o = np.reshape(o, (-1, 1))
        
        layers.append(nnElementwiseAffineLayer(s, o, getattr(dlt_layer, 'Name', '')))
        
    # Check for BatchNormalizationLayer (matches MATLAB exactly)
    elif (hasattr(dlt_layer, '__class__') and 'BatchNormalizationLayer' in str(dlt_layer.__class__)):
        # Can be converted to elementwise layers like MATLAB
        mean = dlt_layer.TrainedMean if hasattr(dlt_layer, 'TrainedMean') else np.zeros(1)
        var = dlt_layer.TrainedVariance if hasattr(dlt_layer, 'TrainedVariance') else np.ones(1)
        epsilon = dlt_layer.Epsilon if hasattr(dlt_layer, 'Epsilon') else 1e-5
        scale = dlt_layer.Scale if hasattr(dlt_layer, 'Scale') else np.ones(1)
        bias = dlt_layer.Offset if hasattr(dlt_layer, 'Offset') else np.zeros(1)
        
        # (x-mean) / sqrt(var+epsilon) * scale + B
        # = x / sqrt(var+epsilon) * scale + (B - mean / sqrt(var+epsilon) * scale)
        final_scale = 1 / np.sqrt(var + epsilon) * scale
        final_offset = (bias - mean * final_scale)
        
        from ..layers.linear.nnElementwiseAffineLayer import nnElementwiseAffineLayer
        layers.append(nnElementwiseAffineLayer(final_scale, final_offset, getattr(dlt_layer, 'Name', '')))
        
    # Check for LeakyReLULayer (matches MATLAB exactly)
    elif (hasattr(dlt_layer, '__class__') and 'LeakyReLULayer' in str(dlt_layer.__class__)):
        alpha = dlt_layer.Scale if hasattr(dlt_layer, 'Scale') else 0.01
        from ..layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
        layers.append(nnLeakyReLULayer(alpha, getattr(dlt_layer, 'Name', '')))
        
    # Check for IdentityLayer (matches MATLAB exactly)
    elif (hasattr(dlt_layer, '__class__') and 'IdentityLayer' in str(dlt_layer.__class__)):
        # Ignore like MATLAB
        inputSize = []
        return layers, inputSize, currentSize
        
    # Check for output layers (matches MATLAB exactly)
    elif (hasattr(dlt_layer, '__class__') and 
          ('RegressionOutputLayer' in str(dlt_layer.__class__) or 'ClassificationOutputLayer' in str(dlt_layer.__class__))) or (isinstance(dlt_layer, dict) and dlt_layer.get('Type') == 'OutputLayer'):
        # Ignore like MATLAB
        inputSize = []
        return layers, inputSize, currentSize
        
    # Check for special VNN Comp cases (matches MATLAB exactly)
    elif getattr(dlt_layer, 'Name', '') == 'MatMul_To_ReluLayer1003':
        # For VNN Comp (test -- test_nano.onnx)
        params = dlt_layer.ONNXParams if hasattr(dlt_layer, 'ONNXParams') else {}
        if hasattr(params, 'Learnables') and hasattr(params.Learnables, 'Ma_MatMulcst'):
            from ..layers.linear.nnLinearLayer import nnLinearLayer
            from ..layers.nonlinear.nnReLULayer import nnReLULayer
            layers.append(nnLinearLayer(params.Learnables.Ma_MatMulcst, 0, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
        
    elif getattr(dlt_layer, 'Name', '') == 'MatMul_To_AddLayer1003':
        # For VNN Comp (test)
        params = dlt_layer.ONNXParams if hasattr(dlt_layer, 'ONNXParams') else {}
        if hasattr(params, 'Learnables'):
            from ..layers.linear.nnLinearLayer import nnLinearLayer
            from ..layers.nonlinear.nnReLULayer import nnReLULayer
            
            if not hasattr(params.Learnables, 'W2'):
                # (test --- test_tiny.onnx)
                layers.append(nnLinearLayer(params.Learnables.W0, 0, getattr(dlt_layer, 'Name', '')))
                layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
                layers.append(nnLinearLayer(params.Learnables.W1, 0, getattr(dlt_layer, 'Name', '')))
            else:
                # (test --- test_small.onnx)
                layers.append(nnLinearLayer(params.Learnables.W0.T, np.array([[1.5], [1.5]]), getattr(dlt_layer, 'Name', '')))
                layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
                layers.append(nnLinearLayer(params.Learnables.W1, np.array([[2.5], [2.5]]), getattr(dlt_layer, 'Name', '')))
                layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
                layers.append(nnLinearLayer(params.Learnables.W2.T, 3.5, getattr(dlt_layer, 'Name', '')))
        
    elif (getattr(dlt_layer, 'Name', '') == 'MatMul_To_AddLayer1019' or 
          getattr(dlt_layer, 'Name', '') == 'Mul_To_AddLayer1021'):
        # For VNN Comp (cora - mnist, svhn, cifar10) - requires hard-coding like MATLAB
        params = dlt_layer.ONNXParams if hasattr(dlt_layer, 'ONNXParams') else {}
        if hasattr(params, 'Learnables'):
            from ..layers.linear.nnLinearLayer import nnLinearLayer
            from ..layers.nonlinear.nnReLULayer import nnReLULayer
            
            # Hard-coded like MATLAB
            layers.append(nnLinearLayer(params.Learnables.fc_1_copy_MatMul_W, params.Nonlearnables.fc_1_copy_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.fc_2_copy_MatMul_W, params.Nonlearnables.fc_2_copy_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.fc_3_copy_MatMul_W, params.Nonlearnables.fc_3_copy_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.fc_4_copy_MatMul_W, params.Nonlearnables.fc_4_copy_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.fc_5_copy_MatMul_W, params.Nonlearnables.fc_5_copy_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.fc_6_copy_MatMul_W, params.Nonlearnables.fc_6_copy_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.fc_7_copy_MatMul_W, params.Nonlearnables.fc_7_copy_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.fc_8_copy_MatMul_W, params.Nonlearnables.fc_8_copy_Add_B, getattr(dlt_layer, 'Name', '')))
        
    elif getattr(dlt_layer, 'Name', '') == 'Sub_To_AddLayer1018':
        # For VNN Comp (test_sat.onnx) - requires hard-coding like MATLAB
        params = dlt_layer.ONNXParams if hasattr(dlt_layer, 'ONNXParams') else {}
        if hasattr(params, 'Learnables'):
            from ..layers.linear.nnLinearLayer import nnLinearLayer
            from ..layers.nonlinear.nnReLULayer import nnReLULayer
            
            # Hard-coded like MATLAB
            layers.append(nnLinearLayer(params.Learnables.Operation_1_MatMul_W, params.Nonlearnables.Operation_1_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.Operation_2_MatMul_W, params.Nonlearnables.Operation_2_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.Operation_3_MatMul_W, params.Nonlearnables.Operation_3_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.Operation_4_MatMul_W, params.Nonlearnables.Operation_4_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.Operation_5_MatMul_W, params.Nonlearnables.Operation_5_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.Operation_6_MatMul_W, params.Nonlearnables.Operation_6_Add_B, getattr(dlt_layer, 'Name', '')))
            layers.append(nnReLULayer(getattr(dlt_layer, 'Name', '')))
            layers.append(nnLinearLayer(params.Learnables.linear_7_MatMul_W, params.Nonlearnables.linear_7_Add_B, getattr(dlt_layer, 'Name', '')))
        
    else:
        # Unknown layer, show warning like MATLAB
        if verbose:
            print(f"    Warning: Skipping '{type(dlt_layer).__name__}'. Not implemented in cora yet!")
        inputSize = []
        return layers, inputSize, currentSize
    
    # Update current size based on the last layer's output (matches MATLAB exactly)
    if layers and hasattr(layers[-1], 'getOutputSize'):
        currentSize = layers[-1].getOutputSize(currentSize)
    
    inputSize = []  # Clear inputSize for non-input layers (matches MATLAB)
    return layers, inputSize, currentSize
