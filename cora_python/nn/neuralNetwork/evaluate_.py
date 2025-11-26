"""
evaluate_ - compute the output of a neural network for the given input
    internal use to speed up computation, use neuralNetwork/evaluate

Syntax:
    res = evaluate_(obj, input, options)
    res = evaluate_(obj, input, options, idxlayer)

Inputs:
    obj - object of class neuralNetwork
    input - input represented as a numeric or set
    options - options for neural network evaluation 
    idxLayer - indices of layers that should be evaluated

Outputs:
    res - evaluation result

Example:
    res = evaluate_(nn, input_data, options)

See also: neuralNetwork/evaluate

Authors:       Tobias Ladner
Written:       21-February-2024
Last update:   21-March-2024 (TL, updateOptions)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

# Import nnHelper methods for proper integration
from cora_python.nn.nnHelper import conversionConZonoStarSet, conversionStarSetConZono

if TYPE_CHECKING:
    from .neuralNetwork import NeuralNetwork


def evaluate_(obj: 'NeuralNetwork', input_data: Any, options: Optional[Dict[str, Any]] = None, 
              idxLayer: Optional[List[int]] = None) -> Any:
    """
    Compute the output of a neural network for the given input.
    Internal use to speed up computation, use neuralNetwork/evaluate
    
    Args:
        obj: object of class neuralNetwork
        input_data: input represented as a numeric or set
        options: options for neural network evaluation
        idxLayer: indices of layers that should be evaluated
        
    Returns:
        res: output of the neural network
    """
    # parse input
    if idxLayer is None:
        # default: all layers
        idxLayer = list(range(len(obj.layers)))  # 0-based indexing like Python
    
    # evaluate ----------------------------------------------------------------
    
    if isinstance(input_data, np.ndarray):  # numeric ---
        r = aux_evaluateNumeric(obj, input_data, options, idxLayer)
    
    elif hasattr(input_data, 'inf') and hasattr(input_data, 'sup'):  # interval ---
        r = aux_evaluateInterval(obj, input_data, options, idxLayer)
    
    elif hasattr(input_data, 'c') and hasattr(input_data, 'G'):  # zonotope/polyZonotope ---
        r = aux_evaluatePolyZonotope(obj, input_data, options, idxLayer)
    
    elif hasattr(input_data, 'coefficients'):  # taylm ---
        r = aux_evaluateTaylm(obj, input_data, options, idxLayer)
    
    elif hasattr(input_data, 'c') and hasattr(input_data, 'A'):  # conZonotope ---
        r = aux_evaluateConZonotope(obj, input_data, options, idxLayer)
    
    else:  # other ---
        raise NotImplementedError(f"Set representation {type(input_data)} is not supported.")
    
    return r


# Auxiliary functions -----------------------------------------------------

def aux_evaluateNumeric(obj: 'NeuralNetwork', input_data: np.ndarray, options: Dict[str, Any], 
                        idxLayer: List[int]) -> np.ndarray:
    """
    Evaluate numeric input.
    
    Args:
        obj: Neural network object
        input_data: Numeric input data
        options: Evaluation options
        idxLayer: Layer indices to evaluate
        
    Returns:
        Evaluation result
    """
    r = input_data
    for k in idxLayer:
        if 'nn' not in options:
            options['nn'] = {}
        options['nn']['layer_k'] = k
        # Use 0-based indexing (Python). MATLAB uses 1-based: layers{k}.
        layer_k = obj.layers[k]
        # Store input for backpropgation
        if options['nn'].get('train', {}).get('backprop', False):
            if not hasattr(layer_k, 'backprop') or not isinstance(layer_k.backprop, dict):
                layer_k.backprop = {'store': {}}
            if 'store' not in layer_k.backprop or not isinstance(layer_k.backprop['store'], dict):
                layer_k.backprop['store'] = {}
            layer_k.backprop['store']['input'] = r
        r = layer_k.evaluateNumeric(r, options)
        options = aux_updateOptions(obj, options, 'numeric', k, layer_k)
    
    return r


def aux_evaluateInterval(obj: 'NeuralNetwork', input_data: Any, options: Dict[str, Any], 
                         idxLayer: List[int]) -> Any:
    """
    Evaluate interval input.
    
    Args:
        obj: Neural network object
        input_data: Interval input data
        options: Evaluation options
        idxLayer: Layer indices to evaluate
        
    Returns:
        Evaluation result
    """
    r = input_data
    for k in idxLayer:
        if 'nn' not in options:
            options['nn'] = {}
        options['nn']['layer_k'] = k
        layer_k = obj.layers[k]  # Python uses 0-based indexing, idxLayer is already 0-based
        # Store input for backpropgation
        if options['nn'].get('train', {}).get('backprop', False):
            if not hasattr(layer_k, 'backprop') or not isinstance(layer_k.backprop, dict):
                layer_k.backprop = {'store': {}}
            if 'store' not in layer_k.backprop or not isinstance(layer_k.backprop['store'], dict):
                layer_k.backprop['store'] = {}
            layer_k.backprop['store']['input'] = r
        r = layer_k.evaluateInterval(r, options)
        options = aux_updateOptions(obj, options, 'interval', k, layer_k)
    
    return r


def aux_evaluatePolyZonotope(obj: 'NeuralNetwork', input_data: Any, options: Dict[str, Any], 
                              idxLayer: List[int]) -> Any:
    """
    Evaluate zonotope/polyZonotope input.
    
    Args:
        obj: Neural network object
        input_data: Zonotope or polyZonotope input data
        options: Evaluation options
        idxLayer: Layer indices to evaluate
        
    Returns:
        Evaluation result
    """
    # we only use polyZonotopes internally
    isZonotope = hasattr(input_data, 'c') and hasattr(input_data, 'G') and not hasattr(input_data, 'E')
    
    if isZonotope:
        # transform to polyZonotope (matches MATLAB exactly)
        # and only use independent generators
        from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
        # Create proper 2D empty arrays to match MATLAB behavior
        n = input_data.c.shape[0]  # number of dimensions
        input_data = PolyZonotope(input_data.c, input_data.G, 
                                 np.zeros((n, 0)),  # GI: n x 0 matrix
                                 np.zeros((0, 0), dtype=int),  # E: 0 x 0 matrix  
                                 np.zeros((0, 1), dtype=int))  # id: 0 x 1 vector
        if 'nn' not in options:
            options['nn'] = {}
        options['nn']['add_approx_error_to_GI'] = True
        options['nn']['remove_GI'] = False
    
    try:
        # prepare propagation
        print(f"DEBUG: aux_evaluatePolyZonotope - input_data.c shape: {input_data.c.shape}")
        print(f"DEBUG: aux_evaluatePolyZonotope - input_data.G shape: {input_data.G.shape}")
        print(f"DEBUG: aux_evaluatePolyZonotope - input_data.GI shape: {input_data.GI.shape}")
        print(f"DEBUG: aux_evaluatePolyZonotope - input_data.E shape: {input_data.E.shape}")
        
        c = input_data.c
        G = input_data.G
        GI = input_data.GI
        E = input_data.E
        id_ = input_data.id
        
        print(f"DEBUG: aux_evaluatePolyZonotope - After extraction: c shape: {c.shape}, G shape: {G.shape}, GI shape: {GI.shape}, E shape: {E.shape}")
        
        # make sure all properties have correct size
        if G.size == 0:
            G = np.zeros((c.shape[0], 0))
            E = np.zeros((0, 0))
            id_ = np.array([])
            id_max = 1
        else:
            id_max = int(np.max(id_)) if id_.size > 0 else 0
            
        if GI.size == 0:
            GI = np.zeros((c.shape[0], 0))
        
        # find all even exponents, also save others
        # (TL: this was done for speed, not sure how important it really is...)
        if E.size > 0:
            ind = np.where(np.prod(np.ones_like(E) - (E % 2), axis=0) == 1)[0]
            ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        else:
            ind = np.array([], dtype=int)
            ind_ = np.array([], dtype=int)
        
        if options.get('nn', {}).get('order_reduction_sensitivity', False):
            # set sensitivity in each layer (used for order reduction)
            obj.calcSensitivity(c)
        
        # iterate over all layers
        for k in idxLayer:
            if 'nn' not in options:
                options['nn'] = {}
            options['nn']['layer_k'] = k
            layer_k = obj.layers[k]  # Python uses 0-based indexing, idxLayer is already 0-based
            print(f"DEBUG: aux_evaluatePolyZonotope - Before layer {k}: c shape: {c.shape}, G shape: {G.shape}, GI shape: {GI.shape}")
            c, G, GI, E, id_, id_max, ind, ind_ = layer_k.evaluatePolyZonotope(
                c, G, GI, E, id_, id_max, ind, ind_, options)
            print(f"DEBUG: aux_evaluatePolyZonotope - After layer {k}: c shape: {c.shape}, G shape: {G.shape}, GI shape: {GI.shape}")
            options = aux_updateOptions(obj, options, 'polyZonotope', k, layer_k)
        
        # build result
        from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
        r = PolyZonotope(c, G, GI, E, id_)
        
    except MemoryError:
        raise MemoryError("Out of memory while processing layer. Try to set options.nn.num_generators to a lower value.")
    
    if isZonotope:
        # transform back to zonotope (matches MATLAB exactly)
        from cora_python.contSet.zonotope.zonotope import Zonotope
        r = Zonotope(r.c, r.G)
    
    return r


def aux_evaluateTaylm(obj: 'NeuralNetwork', input_data: Any, options: Dict[str, Any], 
                       idxLayer: List[int]) -> Any:
    """
    Evaluate Taylor model input.
    
    Args:
        obj: Neural network object
        input_data: Taylor model input data
        options: Evaluation options
        idxLayer: Layer indices to evaluate
        
    Returns:
        Evaluation result
    """
    r = input_data
    for k in idxLayer:
        if 'nn' not in options:
            options['nn'] = {}
        options['nn']['layer_k'] = k
        layer_k = obj.layers[k]  # Python uses 0-based indexing, idxLayer is already 0-based
        r = layer_k.evaluateTaylm(r, options)
        options = aux_updateOptions(obj, options, 'taylm', k, layer_k)
    
    return r


def aux_evaluateConZonotope(obj: 'NeuralNetwork', input_data: Any, options: Dict[str, Any], 
                             idxLayer: List[int]) -> Any:
    """
    Evaluate constrained zonotope input.
    
    Args:
        obj: Neural network object
        input_data: Constrained zonotope input data
        options: Evaluation options
        idxLayer: Layer indices to evaluate
        
    Returns:
        Evaluation result
    """
    # convert constrained zonotope to star set using nnHelper
    c, G, C, d, l, u = conversionConZonoStarSet(input_data)
    
    for k in idxLayer:
        if 'nn' not in options:
            options['nn'] = {}
        options['nn']['layer_k'] = k
        layer_k = obj.layers[k]  # Python uses 0-based indexing, idxLayer is already 0-based
        c, G, C, d, l, u = layer_k.evaluateConZonotope(c, G, C, d, l, u, options)
        options = aux_updateOptions(obj, options, 'conZonotope', k, layer_k)
    
    # convert star set back to constrained zonotope using nnHelper
    r = conversionStarSetConZono(c, G, C, d, l, u)
    
    return r


def aux_updateOptions(obj: 'NeuralNetwork', options: Dict[str, Any], type_: str, k: int, layer_k: Any) -> Dict[str, Any]:
    """
    Update options during evaluation.
    
    Args:
        obj: Neural network object
        options: Current options
        type_: Type of evaluation
        k: Layer index
        layer_k: Layer object
        
    Returns:
        Updated options
    """
    if type_ == 'polyZonotope':
        obj.propagateBounds(k, options)
    
    # Check if it's a GNN projection layer
    if hasattr(layer_k, '__class__') and 'nnGNNProjectionLayer' in str(layer_k.__class__):
        # update graph
        if 'nn' in options and 'graph' in options['nn']:
            options['nn']['graph'] = options['nn']['graph'].subgraph(layer_k.nodes_keep)
            layer_k.updateMessagePassing()
    
    return options
