"""
calcSensitivity - calculate input-output sensitivity matrix

Description:
    Calculate input-output sensitivity matrix at x
    rows correspond to output neurons, columns to input neurons
    sensitivity of layer i will be stored in obj.layers[i].sensitivity

Syntax:
    [S, y] = calcSensitivity(x, options, store_sensitivity)

Inputs:
    x - Input point
    options - Evaluation options
    store_sensitivity - Whether to store sensitivity in layers

Outputs:
    S - Sensitivity matrix
    y - Output at x

Example:
    S, y = nn.calcSensitivity(x, options, True)

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       23-November-2022 (polish)
Last update:   23-November-2022 (polish)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple

# Import nnHelper methods for proper integration
from cora_python.nn.nnHelper import validateNNoptions


def calcSensitivity(self, x: np.ndarray, varargin=None, store_sensitivity: bool = True) -> Tuple[Any, Any]:
    """
    Calculate input-output sensitivity matrix at x
    rows correspond to output neurons, columns correspond to input neurons
    sensitivity of layer i will be stored in obj.layers[i].sensitivity
    
    Args:
        x: Input point
        varargin: Options for neural network evaluation (stored in options.nn)
        store_sensitivity: Whether to store sensitivity in each layer (default: True)
        
    Returns:
        Tuple of (S, y) results
    """
    # parse input - match MATLAB signature
    if varargin is None:
        varargin = {}
    
    # validate options using nnHelper
    options = validateNNoptions(varargin, False)
    
    # Convert numpy input to torch if needed - all internal operations use torch
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    
    device = x.device
    dtype = x.dtype
    
    # forward propagation
    xs = [None] * len(self.layers)
    original_batch_size = x.shape[1] if x.ndim > 1 else 1
    
    for i in range(len(self.layers)):
        xs[i] = x
        layer_i = self.layers[i]
        x = layer_i.evaluateNumeric(x, options)
    
    y = x
    
    # Obtain number of output dimensions and batch size.
    # MATLAB: [nK,bSz] = size(y);
    # This gets the first dimension as nK and second dimension as bSz
    # If y is 2D (nK, bSz), this works correctly
    if y.ndim == 1:
        nK = y.shape[0]
        bSz = 1
    elif y.ndim == 2:
        nK = y.shape[0]
        bSz = y.shape[1]
    else:
        # If y has more dimensions, flatten
        nK = y.shape[0]
        bSz = torch.prod(torch.tensor(y.shape[1:], device=device)).item()
    
    # Initialize the sensitivity in for the output, i.e., identity matrix.
    # MATLAB: S = repmat(eye(nK,'like',y),1,1,bSz)
    # This creates (nK, nK, bSz) - identity matrices for each batch element
    # Use torch.eye and repeat
    S = torch.eye(nK, dtype=dtype, device=device).unsqueeze(2).repeat(1, 1, bSz)
    
    # backward propagation
    for i in range(len(self.layers) - 1, -1, -1):
        layer_i = self.layers[i]
        S = layer_i.evaluateSensitivity(S, xs[i], options)
        # Ensure S maintains 3D shape (nK, input_dim, bSz) throughout backward propagation
        # Some layers might return 2D arrays, so we need to ensure 3D shape
        if S.ndim == 0:
            # S is scalar, this should not happen - reshape to (1, 1, 1)
            S = S.reshape(1, 1, 1)
        elif S.ndim == 1:
            # S is 1D, reshape based on context
            # If it matches input dimension, it's likely (input_dim,), reshape to (1, input_dim, 1)
            # Otherwise, it might be (nK,), reshape to (nK, 1, 1)
            if len(xs) > 0 and xs[i] is not None and S.shape[0] == xs[i].shape[0]:
                S = S.reshape(1, S.shape[0], 1)
            else:
                S = S.reshape(S.shape[0], 1, 1)
        elif S.ndim == 2:
            # S is 2D, add batch dimension: (nK, input_dim) -> (nK, input_dim, 1)
            S = S.reshape(S.shape[0], S.shape[1], 1)
        # If S.ndim == 3, it's already correct shape (nK, input_dim, bSz)
        
        # save sensitivity at layer i for refinement
        if store_sensitivity:
            layer_i.sensitivity = S
    
    return S, y
