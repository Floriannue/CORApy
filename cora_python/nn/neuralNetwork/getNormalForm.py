"""
getNormalForm - transforms the neural network into an identical network 
   in normal form by combining subsequent linear layers s.t. there are 
   only alternating linear and nonlinear layers.
   Note: only works on feed-forward networks

Syntax:
    nn_normal = neuralNetwork.getNormalForm(obj)

Inputs:
    obj - neuralNetwork

Outputs:
    nn_normal - neural network in normal form

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       14-December-2022
Last update:   17-January-2023 (TL, Reshape)
                06-August-2024 (TL, check if already in normal form)
Last revision: 01-August-2023
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, List
from .neuralNetwork import NeuralNetwork
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def getNormalForm(obj: NeuralNetwork) -> NeuralNetwork:
    """
    Transforms the neural network into an identical network in normal form
    
    Args:
        obj: neuralNetwork
        
    Returns:
        nn_normal: neural network in normal form
    """
    layers = []
    
    # check if is already normal form
    if _aux_isNormalForm(obj):
        # create copy - equivalent to cellfun(@(layer) layer.copy(), obj.layers,'UniformOutput',false)
        layers_copy = []
        for layer in obj.layers:
            if hasattr(layer, 'copy'):
                layers_copy.append(layer.copy())
            else:
                layers_copy.append(layer)
        
        nn_normal = NeuralNetwork(layers_copy)
        nn_normal.reset()
        return nn_normal
    
    # init properties
    W = 1
    b = 0
    
    for i in range(len(obj.layers)):
        layer = obj.layers[i]
        
        if hasattr(layer, '__class__') and 'nnConv2DLayer' in str(layer.__class__):
            if hasattr(layer, 'convert2nnLinearLayer'):
                layer = layer.convert2nnLinearLayer()
        
        if hasattr(layer, '__class__') and 'nnLinearLayer' in str(layer.__class__):
            W = layer.W @ W  # equivalent to layer.W * W in MATLAB
            b = layer.W @ b  # equivalent to layer.W * b in MATLAB
            b = np.sum(b, axis=1, keepdims=True)  # equivalent to sum(b, 2) in MATLAB
            b = b + layer.b
        elif hasattr(layer, '__class__') and 'nnElementwiseAffineLayer' in str(layer.__class__):
            W = np.diag(layer.scale) @ W  # equivalent to diag(layer.scale) * W in MATLAB
            b = layer.scale * b + layer.offset  # equivalent to layer.scale .* b + layer.offset in MATLAB
        elif hasattr(layer, '__class__') and 'nnIdentityLayer' in str(layer.__class__):
            # W = W; b = b;
            pass
        elif hasattr(layer, '__class__') and 'nnReshapeLayer' in str(layer.__class__):
            W_reshape = np.eye(np.prod(layer.inputSize))  # equivalent to eye(prod(layer.inputSize)) in MATLAB
            if hasattr(layer, 'evaluateNumeric'):
                W_reshape = layer.evaluateNumeric(W_reshape, {})  # equivalent to struct in MATLAB
            W = W_reshape @ W  # equivalent to W_reshape * W in MATLAB
        elif hasattr(layer, '__class__') and 'nnActivationLayer' in str(layer.__class__):
            # create new linear layer
            linear_layer = _aux_getLinearLayer(W, b)
            if linear_layer is not None:
                layers.append(linear_layer)
            
            # reset properties
            W = 1
            b = 0
            
            # activation
            if hasattr(layer, 'copy'):
                layers.append(layer.copy())
            else:
                layers.append(layer)
        else:
            raise CORAerror('CORA:notSupported',
                           f"Unable to bring a layer of type '{layer.__class__.__name__}' in normal form.")
    
    # Add final linear layer if needed
    final_layer = _aux_getLinearLayer(W, b)
    if final_layer is not None and not _aux_isIdentityLayer(final_layer):
        layers.append(final_layer)
    
    # construct neural network in normal form
    nn_normal = NeuralNetwork(layers)
    
    return nn_normal


def _aux_isNormalForm(nn: NeuralNetwork) -> bool:
    """
    Check if given network is of normal form
    
    Args:
        nn: neural network
        
    Returns:
        bool: True if already in normal form
    """
    # check if given network is of normal form
    for i in range(len(nn.layers)):
        if i % 2 == 0:  # equivalent to mod(i,2) == 1 in MATLAB (0-based indexing)
            if not (hasattr(nn.layers[i], '__class__') and 'nnLinearLayer' in str(nn.layers[i].__class__)):
                return False
        else:
            if not (hasattr(nn.layers[i], '__class__') and 'nnActivationLayer' in str(nn.layers[i].__class__)):
                return False
    
    return True


def _aux_getLinearLayer(W: np.ndarray, b: np.ndarray) -> Any:
    """
    Construct linear layer
    
    Args:
        W: weight matrix
        b: bias vector
        
    Returns:
        layer: constructed layer or None
    """
    # construct linear layer
    if W.size > 1 or b.size > 1:  # equivalent to length(W) > 1 || length(b) > 1 in MATLAB
        from ...nn.layers.linear.nnLinearLayer import nnLinearLayer
        layer = nnLinearLayer(W, b)
        return layer
    elif W != 1 or b != 0:  # equivalent to W ~= 1 || b ~= 0 in MATLAB
        from ...nn.layers.linear.nnElementwiseAffineLayer import nnElementwiseAffineLayer
        layer = nnElementwiseAffineLayer(W, b)
        return layer
    else:
        from ...nn.layers.linear.nnIdentityLayer import nnIdentityLayer
        layer = nnIdentityLayer()
        return layer


def _aux_isIdentityLayer(layer: Any) -> bool:
    """
    Check if layer is an identity layer
    
    Args:
        layer: layer to check
        
    Returns:
        bool: True if identity layer
    """
    return (hasattr(layer, '__class__') and 
            'nnIdentityLayer' in str(layer.__class__))
