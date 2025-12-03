"""
castWeights - cast learnable weights of a neural network

Syntax:
    nn.castWeights(x, idxLayer)

Inputs:
    nn - object of class neuralNetwork
    x - instance of data type
    idxLayer - indices of layers that should be evaluated

Outputs:
    -

References:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Lukas Koller
Written:       04-December-2023
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, List, Optional, Union
from .neuralNetwork import NeuralNetwork


def castWeights(nn: NeuralNetwork, x: Union[np.ndarray, np.dtype], idxLayer: Optional[List[int]] = None) -> None:
    """
    Cast learnable weights of a neural network
    
    Args:
        nn: NeuralNetwork object
        x: instance of data type or numpy dtype
        idxLayer: indices of layers that should be evaluated (defaults to all layers)
    """
    # validate parameters
    if idxLayer is None:
        # 1-based indexing like MATLAB
        idxLayer = list(range(len(nn.layers)))  # 0-based indexing like Python
    
    # Determine the target dtype
    print(f"DEBUG: x type: {type(x)}, x value: {x}")
    if hasattr(x, 'dtype') and not isinstance(x, type):
        # x is a numpy array (instance), use its dtype
        target_dtype = x.dtype
        print(f"DEBUG: x has dtype attribute: {target_dtype}")
    elif isinstance(x, np.dtype):
        # x is already a numpy dtype
        target_dtype = x
        print(f"DEBUG: x is numpy dtype: {target_dtype}")
    elif isinstance(x, type):
        # x is a class (like numpy.float32), create an instance to get dtype
        try:
            target_dtype = np.dtype(x)
            print(f"DEBUG: x is class, created dtype: {target_dtype}")
        except:
            # Fallback to float64 (MATLAB uses double precision)
            target_dtype = np.float64
            print(f"DEBUG: fallback to float64: {target_dtype}")
    else:
        # x is some other type, try to convert to numpy dtype
        try:
            target_dtype = np.dtype(type(x))
            print(f"DEBUG: converted x to dtype: {target_dtype}")
        except:
            # Fallback to float64 (MATLAB uses double precision)
            target_dtype = np.float64
            print(f"DEBUG: fallback to float64: {target_dtype}")
    
    print(f"DEBUG: Final target_dtype: {target_dtype}, type: {type(target_dtype)}")

    for i in idxLayer:
        layeri = nn.layers[i]  # Convert to 0-based indexing
        # move all learnable parameters to gpu
        names = layeri.getLearnableParamNames()
        for j in range(len(names)):
            # cast learnable weights
            if hasattr(layeri, names[j]):
                current_value = getattr(layeri, names[j])
                # Cast to target dtype (equivalent to MATLAB's cast(..., 'like', x))
                try:
                    if hasattr(current_value, 'astype'):
                        # numpy array or similar
                        setattr(layeri, names[j], current_value.astype(target_dtype))
                    elif isinstance(current_value, (int, float, np.number)):
                        # scalar values
                        setattr(layeri, names[j], target_dtype.type(current_value))
                    else:
                        # other types, try to convert
                        setattr(layeri, names[j], target_dtype.type(current_value))
                except (TypeError, ValueError) as e:
                    # Log the error instead of silently failing
                    print(f"Warning: Failed to cast parameter '{names[j]}' in layer {i} from type {type(current_value)} to {target_dtype}: {e}")
                    print(f"  Parameter value: {current_value}")
                    # Keep original value but log the issue
                    continue
        
        # Notify layer
        if hasattr(layeri, 'castWeights'):
            layeri.castWeights(target_dtype)
