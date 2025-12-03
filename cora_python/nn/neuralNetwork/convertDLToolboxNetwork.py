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


def convertDLToolboxNetwork(layer_dicts: List, verbose: bool = False) -> NeuralNetwork:
    """
    Convert intermediate layer representation (from ONNX parsing) to CORA network.
    
    This function converts layer dictionaries created from ONNX parsing into
    CORA layer objects. The function name is kept for API compatibility with MATLAB.
    
    Args:
        layer_dicts: List of layer dictionaries (from ONNX parsing)
        verbose: Whether to print verbose output
        
    Returns:
        CORA neural network
    """
    n = len(layer_dicts)
    
    layers = []
    inputSize = []
    currentSize = []
    
    if verbose:
        print("Converting onnx representation to CORA neuralNetwork...")
    
    i = 0
    while i < n:
        layer_dict = layer_dicts[i]
        if verbose:
            layer_type = layer_dict.get('Type', 'Unknown') if isinstance(layer_dict, dict) else 'Unknown'
            print(f"#{i+1}: {layer_type}")
        
        # All layers are dicts (flat list). Convert directly in order
        layers, inputSize_, currentSize = aux_convertLayer(layers, layer_dict, currentSize, verbose)
        if not inputSize:
            inputSize = inputSize_
        
        i += 1
    
    # instantiate neural network
    obj = NeuralNetwork(layers)
    
    # Set input size from inputSize variable (matches MATLAB exactly)
    if inputSize:
        if np.isscalar(inputSize):
            inputSize = [inputSize, 1]
        
        # For convolutional networks, preserve spatial dimensions [H, W, C]
        # For fully connected networks, flatten to [features, 1]
        # If inputSize has 3 or 4 dimensions, it's a convolutional network
        if len(inputSize) == 3 or len(inputSize) == 4:
            # Convolutional network: keep spatial dimensions [H, W, C] or [H, W, C, batch]
            final_input_size = inputSize
        else:
            # Fully connected network: flatten to [features, 1]
            final_input_size = [np.prod(inputSize), 1] if len(inputSize) > 1 else inputSize
        
        if verbose:
            print(f"Setting network input size: {final_input_size}")
        obj.setInputSize(final_input_size, False)
    
    return obj


def aux_convertLayer(layers: List, layer_dict: Dict, currentSize: List, verbose: bool) -> Tuple[List, List, List]:
    """
    Convert a single layer dictionary to CORA layer.
    
    Args:
        layers: Current list of CORA layers
        layer_dict: Layer dictionary to convert (from ONNX parsing)
        currentSize: Current input size
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (layers, inputSize, currentSize)
    """
    inputSize = []
    
    # Validate layer is a non-empty dict
    if not layer_dict or not isinstance(layer_dict, dict) or not layer_dict:
        if verbose:
            print("    Warning: Empty or invalid layer, skipping")
        return layers, inputSize, currentSize
    
    # Get layer type for logging
    layer_type = layer_dict.get('Type', 'Unknown')
    if verbose:
        print(f"  Converting layer type: {layer_type}")
    
    # Check for InputLayer (matches MATLAB exactly)
    if layer_dict.get('Type') == 'InputLayer':
        # Input layer - extract input size (matches MATLAB exactly)
        inputSize = layer_dict.get('InputSize', [1, 1])
        if len(inputSize) == 1:
            inputSize = [inputSize[0], 1]
        elif len(inputSize) == 3:
            # channel dimension should be last: [h,w,c] (like MATLAB)
            pass
        elif len(inputSize) == 4:
            # 4D input like [1, 1, 1, 5] - keep as is (like MATLAB)
            pass
            
        if verbose:
            print(f"    Found input size: {inputSize}")
        
        # Handle normalization like MATLAB (if present)
        if layer_dict.get('Normalization') == 'zscore':
            mu = layer_dict.get('Mean', np.zeros(1, dtype=np.float64))
            sigma = layer_dict.get('StandardDeviation', np.ones(1, dtype=np.float64))
            from ..layers.linear.nnElementwiseAffineLayer import nnElementwiseAffineLayer
            # Add normalization layer to computational layers (like MATLAB)
            layers.append(nnElementwiseAffineLayer(1/sigma, -mu/sigma, layer_dict.get('Name', '')))
        
        # Input layer itself is just metadata, don't add to computational layers
        # Only normalization layers (if any) are added above
        # The currentSize should be the original input size - reshape layers handle flattening
        currentSize = inputSize
            
        return layers, inputSize, currentSize
    
    # Check for FullyConnectedLayer (matches MATLAB exactly)
    elif layer_dict.get('Type') == 'FullyConnectedLayer':
        # Linear layer (matches MATLAB exactly)
        from ..layers.linear.nnLinearLayer import nnLinearLayer

        # Extract weights and biases
        W = layer_dict.get('Weight', np.eye(1))
        b = layer_dict.get('Bias', np.zeros((1, 1)))
        
        # Ensure float64 precision (weights from ONNX may be float32)
        W = np.asarray(W, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # ONNX MatMul stores weights as [in_features, out_features] -> needs transpose
        # ONNX Gemm stores weights as [out_features, in_features] (after handling transB) -> no transpose needed
        # CORA needs W as [out_features, in_features] for W @ x
        if layer_dict.get('FromGemm', False):
            # Gemm weights are already in [out_features, in_features] format (after transB handling in readONNXNetwork)
            # No transpose needed
            pass
        elif layer_dict.get('FromMatMul', False):
            # MatMul weights are [in_features, out_features], transpose to [out_features, in_features]
            W = W.T
        else:
            # Default: assume MatMul format and transpose
            W = W.T

        # Ensure proper shapes
        if len(W.shape) == 1:
            W = W.reshape(-1, 1)
        if len(b.shape) == 1:
            b = b.reshape(-1, 1)
        
        # After transpose, W.shape[0] = out_features
        # Bias should have shape [out_features, 1]
        # If bias shape doesn't match (e.g., it was transposed incorrectly), fix it
        if b.shape[0] != W.shape[0]:
            # This shouldn't happen if ONNX format is correct, but handle it
            if b.shape[0] == W.shape[1]:
                # Bias matches input features instead of output features - this is wrong
                # Create a zero bias of the correct size
                if verbose:
                    print(f"Warning: Bias shape {b.shape} doesn't match W.shape[0]={W.shape[0]}. Using zero bias.")
                b = np.zeros((W.shape[0], 1), dtype=np.float64)

        layer = nnLinearLayer(W, b, name=layer_dict.get('Name', ''))
        layers.append(layer)

        # Update sizes (matches MATLAB exactly)
        if not currentSize:
            currentSize = [W.shape[1], 1]  # input features from weights
        currentSize = [W.shape[0], 1]  # output features
        
    # Elementwise affine layers (offset/scale) from ONNX Sub/Add/Mul fusion
    elif layer_dict.get('Type') == 'ElementwiseAffineLayer':
        from ..layers.linear.nnElementwiseAffineLayer import nnElementwiseAffineLayer
        scale = layer_dict.get('Scale', 1)
        offset = layer_dict.get('Offset', 0)
        # Follow MATLAB strictly: use given tensors as-is but flatten feature dimension
        scale = np.array(scale).astype(float).reshape(-1, 1)
        offset = np.array(offset).astype(float).reshape(-1, 1)
        # If they are scalar, shapes become (1,1). Layer will broadcast with input
        layer = nnElementwiseAffineLayer(scale, offset, name=layer_dict.get('Name', ''))
        layers.append(layer)
        # size does not change
        if not currentSize:
            currentSize = [scale.shape[0], 1]
        
    # Check for ReLULayer (matches MATLAB exactly)
    elif layer_dict.get('Type') == 'ReLULayer':
        # ReLU activation layer
        from ..layers.nonlinear.nnReLULayer import nnReLULayer
        layer = nnReLULayer(name=layer_dict.get('Name', ''))
        layers.append(layer)
        
    # Check for SigmoidLayer (matches MATLAB exactly)
    elif layer_dict.get('Type') == 'SigmoidLayer':
        # Sigmoid activation layer
        from ..layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
        layer = nnSigmoidLayer(name=layer_dict.get('Name', ''))
        layers.append(layer)
        
    # Check for TanhLayer (matches MATLAB exactly)
    elif layer_dict.get('Type') == 'TanhLayer':
        # Tanh activation layer
        from ..layers.nonlinear.nnTanhLayer import nnTanhLayer
        layer = nnTanhLayer(name=layer_dict.get('Name', ''))
        layers.append(layer)
        
    # Check for reshape layers (matches MATLAB exactly)
    elif layer_dict.get('Type') == 'ReshapeLayer':
        # Reshape layer
        from ..layers.other.nnReshapeLayer import nnReshapeLayer
        
        # Extract shape information consistently
        shape = layer_dict.get('Shape', [-1])  # Default to flatten if not specified
        
        if verbose:
            print(f"    DEBUG ReshapeLayer: shape from layer_dict: {shape}, currentSize: {currentSize}, layer_dict keys: {list(layer_dict.keys())}")
        
        # Special handling for Flatten layers (which were converted to ReshapeLayer)
        if 'FlattenAxis' in layer_dict:
            # This was originally a Flatten layer
            flatten_axis = layer_dict['FlattenAxis']
            if verbose:
                print(f"    Converting Flatten layer with axis {flatten_axis}")
            
            # Check if Shape was already set from ONNX Reshape operation
            if 'Shape' in layer_dict and layer_dict['Shape'] != [-1]:
                # Use the shape from ONNX Reshape operation
                shape = layer_dict['Shape']
                if verbose:
                    print(f"    Using Shape from ONNX: {shape}")
            else:
                # For Flatten layers, use [-1] to indicate flattening to 1D
                # The ReshapeLayer will handle the flattening correctly
                shape = [-1]
                if verbose:
                    print(f"    Using [-1] for Flatten layer (flatten to 1D)")
        
        # Ensure the shape is valid for the ReshapeLayer
        if shape and -1 not in shape:
            # We have a concrete shape, use it
            pass
        elif shape == [-1]:
            # Dynamic shape - this means flatten to 1D
            # For the network to work properly, we need to ensure the ReshapeLayer
            # can handle the input transformation correctly
            if verbose:
                print(f"    Using dynamic shape [-1] for ReshapeLayer")
        else:
            # Fallback to flattening
            shape = [-1]
            if verbose:
                print(f"    Fallback to flattening with shape: {shape}")
        
        layer = nnReshapeLayer(shape, name=layer_dict.get('Name', ''))
        layers.append(layer)
        
        # Update size based on reshape
        # For ReshapeLayer with [-1], it flattens to 1D, so output size is [total_input_size, 1]
        if shape == [-1]:
            # Flatten operation - output size is the total flattened input size
            input_size = currentSize.copy() if currentSize else []
            if input_size:
                currentSize = [np.prod(input_size), 1]
                if verbose:
                    print(f"    DEBUG: Flatten with [-1] - input: {input_size} -> currentSize: {currentSize}")
            else:
                if verbose:
                    print(f"    DEBUG: Flatten with [-1] but no currentSize available")
        elif shape and -1 not in shape:
            # ReshapeLayer with indices - output size is number of indices
            # But indices are for reordering, so output size should match input size
            # Actually, ReshapeLayer.getOutputSize returns [num_indices, 1]
            input_size = currentSize.copy() if currentSize else []
            num_indices = len(shape) if isinstance(shape, list) else np.prod(shape) if hasattr(shape, 'size') else 1
            # For index-based reshape, output size should be based on the number of unique indices
            # But typically it's the same as input size (just reordered)
            if input_size:
                # Keep the same total size (just reordered)
                currentSize = [np.prod(input_size), 1]
            else:
                currentSize = [num_indices, 1]
            if verbose:
                print(f"    DEBUG: Reshape with indices - input: {input_size}, num_indices: {num_indices} -> currentSize: {currentSize}")
        else:
            if verbose:
                print(f"    DEBUG: No size update - shape: {shape}, currentSize: {currentSize}")
        
    # Check for Convolution2DLayer
    elif layer_dict.get('Type') == 'Conv2DLayer':
        # Convolutional layer
        from ..layers.linear.nnConv2DLayer import nnConv2DLayer
        
        # Extract convolution parameters
        W = layer_dict.get('Weight', np.eye(1, dtype=np.float64))
        b = layer_dict.get('Bias', np.zeros(1, dtype=np.float64))
        padding = layer_dict.get('Padding', [0, 0, 0, 0])  # ONNX pads format: [pad_top, pad_left, pad_bottom, pad_right]
        
        # ONNX Conv weights are in format: [out_channels, in_channels, kernel_height, kernel_width]
        # MATLAB nnConv2DLayer expects: [kernel_height, kernel_width, in_channels, out_channels]
        # Transpose from ONNX to MATLAB format
        if len(W.shape) == 4:
            W = np.transpose(W, (2, 3, 1, 0))  # [out, in, H, W] -> [H, W, in, out]
        
        # MATLAB nnConv2DLayer expects padding as [left, top, right, bottom] (1D array)
        # ONNX provides pads as [top, left, bottom, right] for 2D convolution
        # Convert ONNX format to MATLAB format
        if len(padding) == 4:
            # ONNX: [top, left, bottom, right] -> MATLAB: [left, top, right, bottom]
            padding = np.array([padding[1], padding[0], padding[3], padding[2]])
        elif len(padding) == 2:
            # If only 2 values, assume [height_pad, width_pad]
            padding = np.array([padding[1], padding[0], padding[1], padding[0]])
        else:
            padding = np.array([0, 0, 0, 0])
        
        stride = layer_dict.get('Stride', [1, 1])
        dilation = layer_dict.get('Dilation', [1, 1])
        
        # Ensure arrays are numpy arrays
        stride = np.array(stride) if not isinstance(stride, np.ndarray) else stride
        dilation = np.array(dilation) if not isinstance(dilation, np.ndarray) else dilation
        
        # Ensure bias is a 1D array with correct size (matches number of output channels)
        b = np.asarray(b)
        if b.ndim > 1:
            b = b.flatten()
        # After transpose, W.shape[3] is the number of output channels
        expected_bias_size = W.shape[3] if len(W.shape) == 4 else 1
        if b.size != expected_bias_size:
            if verbose:
                print(f"  WARNING: Bias size {b.size} doesn't match expected {expected_bias_size}. Using zeros.")
            b = np.zeros(expected_bias_size, dtype=np.float64)
        
        layer = nnConv2DLayer(W, b, padding, stride, dilation, name=layer_dict.get('Name', ''))
        layers.append(layer)
        
    # Check for pooling layers
    elif layer_dict.get('Type') == 'PoolingLayer':
        pool_type = layer_dict.get('PoolType', 'AveragePool')
        if pool_type == 'MaxPool':
            from ..layers.nonlinear.nnMaxPool2DLayer import nnMaxPool2DLayer
            poolSize = layer_dict.get('KernelSize', [2, 2])
            stride = layer_dict.get('Stride', [1, 1])
            poolSize = np.array(poolSize) if not isinstance(poolSize, np.ndarray) else poolSize
            stride = np.array(stride) if not isinstance(stride, np.ndarray) else stride
            layer = nnMaxPool2DLayer(poolSize, stride, name=layer_dict.get('Name', ''))
        else:  # AveragePool
            from ..layers.linear.nnAvgPool2DLayer import nnAvgPool2DLayer
            poolSize = layer_dict.get('KernelSize', [2, 2])
            padding = layer_dict.get('Padding', [0, 0, 0, 0])
            
            # MATLAB nnAvgPoolLayer expects padding as [left, top, right, bottom] (1D array)
            # Convert ONNX format to MATLAB format (same as Conv2D)
            if len(padding) == 4:
                # ONNX: [top, left, bottom, right] -> MATLAB: [left, top, right, bottom]
                padding = np.array([padding[1], padding[0], padding[3], padding[2]])
            elif len(padding) == 2:
                # If only 2 values, assume [height_pad, width_pad]
                padding = np.array([padding[1], padding[0], padding[1], padding[0]])
            else:
                padding = np.array([0, 0, 0, 0])
            
            # Default stride should match poolSize (like nnAvgPool2DLayer constructor)
            stride = layer_dict.get('Stride', None)
            if stride is None:
                stride = poolSize  # Default to poolSize like nnAvgPool2DLayer
            stride = np.array(stride) if not isinstance(stride, np.ndarray) else stride
            dilation = np.array([1, 1])  # Default like MATLAB
            poolSize = np.array(poolSize) if not isinstance(poolSize, np.ndarray) else poolSize
            
            layer = nnAvgPool2DLayer(poolSize, padding, stride, dilation, name=layer_dict.get('Name', ''))
        
        layers.append(layer)
        
    # Check for BatchNormalizationLayer
    elif layer_dict.get('Type') == 'BatchNormLayer':
        # Can be converted to elementwise layers like MATLAB
        mean = layer_dict.get('Mean', np.zeros(1, dtype=np.float64))
        var = layer_dict.get('Variance', np.ones(1, dtype=np.float64))
        epsilon = layer_dict.get('Epsilon', 1e-5)
        scale = layer_dict.get('Scale', np.ones(1, dtype=np.float64))
        bias = layer_dict.get('Bias', np.zeros(1, dtype=np.float64))
        
        # (x-mean) / sqrt(var+epsilon) * scale + B
        # = x / sqrt(var+epsilon) * scale + (B - mean / sqrt(var+epsilon) * scale)
        final_scale = 1 / np.sqrt(var + epsilon) * scale
        final_offset = (bias - mean * final_scale)
        
        from ..layers.linear.nnElementwiseAffineLayer import nnElementwiseAffineLayer
        layers.append(nnElementwiseAffineLayer(final_scale, final_offset, layer_dict.get('Name', '')))
        
    # Check for SoftmaxLayer
    elif layer_dict.get('Type') == 'SoftmaxLayer':
        from ..layers.nonlinear.nnSoftmaxLayer import nnSoftmaxLayer
        layer = nnSoftmaxLayer(name=layer_dict.get('Name', ''))
        layers.append(layer)
        
    # Check for LeakyReLULayer
    elif layer_dict.get('Type') == 'LeakyReLULayer':
        alpha = layer_dict.get('Alpha', 0.01)
        from ..layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
        layers.append(nnLeakyReLULayer(alpha, layer_dict.get('Name', '')))
        
    # Check for IdentityLayer
    elif layer_dict.get('Type') == 'IdentityLayer':
        # Ignore like MATLAB
        inputSize = []
        return layers, inputSize, currentSize
        
    # Check for output layers
    elif layer_dict.get('Type') == 'OutputLayer':
        # Ignore like MATLAB
        inputSize = []
        return layers, inputSize, currentSize
        
    # Check for special VNN Comp cases (matches MATLAB exactly)
    elif layer_dict.get('Name') == 'MatMul_To_ReluLayer1003':
        # For VNN Comp (test -- test_nano.onnx)
        params = layer_dict.get('ONNXParams', {})
        if 'Learnables' in params and 'Ma_MatMulcst' in params['Learnables']:
            from ..layers.linear.nnLinearLayer import nnLinearLayer
            from ..layers.nonlinear.nnReLULayer import nnReLULayer
            layers.append(nnLinearLayer(params['Learnables']['Ma_MatMulcst'], 0, layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
        
    elif layer_dict.get('Name') == 'MatMul_To_AddLayer1003':
        # For VNN Comp (test)
        params = layer_dict.get('ONNXParams', {})
        if 'Learnables' in params:
            from ..layers.linear.nnLinearLayer import nnLinearLayer
            from ..layers.nonlinear.nnReLULayer import nnReLULayer
            
            if 'W2' not in params['Learnables']:
                # (test --- test_tiny.onnx)
                layers.append(nnLinearLayer(params['Learnables']['W0'], 0, layer_dict.get('Name', '')))
                layers.append(nnReLULayer(layer_dict.get('Name', '')))
                layers.append(nnLinearLayer(params['Learnables']['W1'], 0, layer_dict.get('Name', '')))
            else:
                # (test --- test_small.onnx)
                layers.append(nnLinearLayer(params['Learnables']['W0'].T, np.array([[1.5], [1.5]]), layer_dict.get('Name', '')))
                layers.append(nnReLULayer(layer_dict.get('Name', '')))
                layers.append(nnLinearLayer(params['Learnables']['W1'], np.array([[2.5], [2.5]]), layer_dict.get('Name', '')))
                layers.append(nnReLULayer(layer_dict.get('Name', '')))
                layers.append(nnLinearLayer(params['Learnables']['W2'].T, 3.5, layer_dict.get('Name', '')))
        
    elif (layer_dict.get('Name') == 'MatMul_To_AddLayer1019' or 
          layer_dict.get('Name') == 'Mul_To_AddLayer1021'):
        # For VNN Comp (cora - mnist, svhn, cifar10) - requires hard-coding like MATLAB
        params = layer_dict.get('ONNXParams', {})
        if 'Learnables' in params:
            from ..layers.linear.nnLinearLayer import nnLinearLayer
            from ..layers.nonlinear.nnReLULayer import nnReLULayer
            
            # Hard-coded like MATLAB
            layers.append(nnLinearLayer(params['Learnables']['fc_1_copy_MatMul_W'], params['Nonlearnables']['fc_1_copy_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['fc_2_copy_MatMul_W'], params['Nonlearnables']['fc_2_copy_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['fc_3_copy_MatMul_W'], params['Nonlearnables']['fc_3_copy_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['fc_4_copy_MatMul_W'], params['Nonlearnables']['fc_4_copy_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['fc_5_copy_MatMul_W'], params['Nonlearnables']['fc_5_copy_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['fc_6_copy_MatMul_W'], params['Nonlearnables']['fc_6_copy_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['fc_7_copy_MatMul_W'], params['Nonlearnables']['fc_7_copy_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['fc_8_copy_MatMul_W'], params['Nonlearnables']['fc_8_copy_Add_B'], layer_dict.get('Name', '')))
        
    elif layer_dict.get('Name') == 'Sub_To_AddLayer1018':
        # For VNN Comp (test_sat.onnx) - requires hard-coding like MATLAB
        params = layer_dict.get('ONNXParams', {})
        if 'Learnables' in params:
            from ..layers.linear.nnLinearLayer import nnLinearLayer
            from ..layers.nonlinear.nnReLULayer import nnReLULayer
            
            # Hard-coded like MATLAB
            layers.append(nnLinearLayer(params['Learnables']['Operation_1_MatMul_W'], params['Nonlearnables']['Operation_1_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['Operation_2_MatMul_W'], params['Nonlearnables']['Operation_2_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['Operation_3_MatMul_W'], params['Nonlearnables']['Operation_3_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['Operation_4_MatMul_W'], params['Nonlearnables']['Operation_4_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['Operation_5_MatMul_W'], params['Nonlearnables']['Operation_5_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['Operation_6_MatMul_W'], params['Nonlearnables']['Operation_6_Add_B'], layer_dict.get('Name', '')))
            layers.append(nnReLULayer(layer_dict.get('Name', '')))
            layers.append(nnLinearLayer(params['Learnables']['linear_7_MatMul_W'], params['Nonlearnables']['linear_7_Add_B'], layer_dict.get('Name', '')))
        
    else:
        # Unknown layer, show warning like MATLAB
        if verbose:
            layer_name = layer_dict.get('Type', 'Unknown') if isinstance(layer_dict, dict) else 'Unknown'
            print(f"    Warning: Skipping '{layer_name}'. Not implemented in cora yet!")
        inputSize = []
        return layers, inputSize, currentSize
    
    # Update current size based on the last layer's output (matches MATLAB exactly)
    if layers and hasattr(layers[-1], 'getOutputSize'):
        currentSize = layers[-1].getOutputSize(currentSize)
    
    inputSize = []  # Clear inputSize for non-input layers (matches MATLAB)
    return layers, inputSize, currentSize
