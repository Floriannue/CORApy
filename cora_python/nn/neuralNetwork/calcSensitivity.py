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
    
    # forward propagation
    xs = [None] * len(self.layers)
    for i in range(len(self.layers)):
        xs[i] = x
        layer_i = self.layers[i]
        x = layer_i.evaluateNumeric(x, options)
    
    y = x
    
    # Obtain number of output dimensions and batch size.
    nK, bSz = y.shape
    
    # Initialize the sensitivity in for the output, i.e., identity matrix.
    # MATLAB: S = repmat(eye(nK,'like',y),1,1,bSz)
    # This creates (nK, nK, bSz) - identity matrices for each batch element
    S = np.tile(np.eye(nK, dtype=y.dtype).reshape(nK, nK, 1), (1, 1, bSz))
    
    # backward propagation
    for i in range(len(self.layers) - 1, -1, -1):
        layer_i = self.layers[i]
        S = layer_i.evaluateSensitivity(S, xs[i], options)
        # save sensitivity at layer i for refinement
        if store_sensitivity:
            layer_i.sensitivity = S
    
    return S, y
