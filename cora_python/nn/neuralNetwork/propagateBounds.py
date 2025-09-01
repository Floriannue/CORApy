"""
propagateBounds - propagates the bounds of layer i to the next activation
   layer and updates the bounds accordingly

Syntax:
    neuralNetwork.propagateBounds(obj, i)
    neuralNetwork.propagateBounds(obj, i, options)

Inputs:
    obj - neuralNetwork
    i - starting layer
    options - struct, evaluation parameters (stored in options.nn)

Outputs:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/evaluate

Authors:       Tobias Ladner
Written:       15-December-2022
Last update:   ---
                ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .neuralNetwork import NeuralNetwork


def propagateBounds(obj: 'NeuralNetwork', i: int, options: Dict[str, Any]):
    """
    Propagate bounds of layer i to the next activation layer and update bounds accordingly.
    
    Args:
        obj: neuralNetwork object
        i: starting layer index
        options: evaluation parameters (stored in options.nn)
    """
    # parse input
    if not options.get('nn', {}).get('propagate_bounds', False) or \
       not options.get('nn', {}).get('reuse_bounds', False):
        return
    
    layer_i = obj.layers[i]
    if not hasattr(layer_i, 'l') or not hasattr(layer_i, 'u'):
        return
    
    l = layer_i.l
    u = layer_i.u
    
    if l is None or u is None or np.any(np.isnan(l)) or np.any(np.isnan(u)):
        return
    
    # Create interval from bounds
    from cora_python.contSet.interval.interval import Interval
    bounds = Interval(l, u)
    bounds = layer_i.evaluateInterval(bounds, options)
    
    # Propagate bounds
    for j in range(i + 1, len(obj.layers)):
        layer_j = obj.layers[j]
        
        if hasattr(layer_j, 'l') and hasattr(layer_j, 'u'):
            if layer_j.l is None or np.any(np.isnan(layer_j.l)) or \
               layer_j.u is None or np.any(np.isnan(layer_j.u)):
                # Set bounds
                layer_j.l = bounds.inf
                layer_j.u = bounds.sup
            else:
                # Update bounds
                Ilayer = Interval(layer_j.l, layer_j.u)
                bounds = bounds & Ilayer
                layer_j.l = bounds.inf
                layer_j.u = bounds.sup
            return
        
        bounds = layer_j.evaluateInterval(bounds, options)
