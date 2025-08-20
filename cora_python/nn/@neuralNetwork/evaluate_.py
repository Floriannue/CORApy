"""
evaluate_ - internal evaluation method

Description:
    Internal evaluation method - compute the output of a neural network for the given input
    internal use to speed up computation, use neuralNetwork/evaluate

Syntax:
    r = evaluate_(input_data, options, idxLayer)

Inputs:
    input_data - Input data or set
    options - Evaluation options
    idxLayer - Indices of layers to evaluate

Outputs:
    r - Evaluation result

Example:
    r = nn.evaluate_(x, options, [0, 1, 2])

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       23-November-2022 (polish)
Last update:   23-November-2022 (polish)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional

def evaluate_(self, input_data: Any, options: Optional[Dict[str, Any]] = None, 
              idxLayer: Optional[List[int]] = None) -> Any:
    """
    Internal evaluation method - compute the output of a neural network for the given input
    internal use to speed up computation, use neuralNetwork/evaluate
    
    Args:
        input_data: Input data or set
        options: Evaluation options
        idxLayer: Indices of layers to evaluate
        
    Returns:
        r: Evaluation result
    """
    if options is None:
        options = {}
    
    # parse input
    if idxLayer is None:
        # default: all layers
        idxLayer = list(range(len(self.layers)))
    
    # evaluate based on input type
    if isinstance(input_data, np.ndarray):  # numeric
        r = self._aux_evaluateNumeric(input_data, options, idxLayer)
    elif hasattr(input_data, 'inf') and hasattr(input_data, 'sup'):  # interval
        r = self._aux_evaluateInterval(input_data, options, idxLayer)
    elif hasattr(input_data, 'center') and hasattr(input_data, 'generators'):  # zonotope/polyZonotope
        r = self._aux_evaluatePolyZonotope(input_data, options, idxLayer)
    elif hasattr(input_data, 'monomials'):  # taylm
        r = self._aux_evaluateTaylm(input_data, options, idxLayer)
    elif hasattr(input_data, 'C') and hasattr(input_data, 'd'):  # conZonotope
        r = self._aux_evaluateConZonotope(input_data, options, idxLayer)
    else:
        raise NotImplementedError(f"Set representation {type(input_data)} is not supported.")
    
    return r

def _aux_evaluateNumeric(self, input_data: np.ndarray, options: Dict[str, Any], idxLayer: List[int]) -> np.ndarray:
    """Evaluate numeric input"""
    r = input_data
    for k in idxLayer:
        if 'nn' not in options:
            options['nn'] = {}
        options['nn']['layer_k'] = k
        layer_k = self.layers[k]
        
        # Store input for backpropagation
        if options.get('nn', {}).get('train', {}).get('backprop', False):
            if not hasattr(layer_k, 'backprop'):
                layer_k.backprop = {}
            if 'store' not in layer_k.backprop:
                layer_k.backprop['store'] = {}
            layer_k.backprop['store']['input'] = r
        
        r = layer_k.evaluateNumeric(r, options)
        options = self._aux_updateOptions(options, 'numeric', k, layer_k)
    
    return r

def _aux_evaluateInterval(self, input_data: Any, options: Dict[str, Any], idxLayer: List[int]) -> Any:
    """Evaluate interval input"""
    r = input_data
    for k in idxLayer:
        if 'nn' not in options:
            options['nn'] = {}
        options['nn']['layer_k'] = k
        layer_k = self.layers[k]
        
        # Store input for backpropagation
        if options.get('nn', {}).get('train', {}).get('backprop', False):
            if not hasattr(layer_k, 'backprop'):
                layer_k.backprop = {}
            if 'store' not in layer_k.backprop:
                layer_k.backprop['store'] = {}
            layer_k.backprop['store']['input'] = r
        
        r = layer_k.evaluateInterval(r, options)
        options = self._aux_updateOptions(options, 'interval', k, layer_k)
    
    return r

def _aux_evaluatePolyZonotope(self, input_data: Any, options: Dict[str, Any], idxLayer: List[int]) -> Any:
    """Evaluate zonotope/polyZonotope input"""
    # we only use polyZonotopes internally
    isZonotope = hasattr(input_data, 'center') and hasattr(input_data, 'generators') and not hasattr(input_data, 'E')
    
    if isZonotope:
        # transform to polyZonotope and only use independent generators
        # This would require PolyZonotope import
        # For now, we'll work with the original input
        if 'nn' not in options:
            options['nn'] = {}
        options['nn']['add_approx_error_to_GI'] = True
        options['nn']['remove_GI'] = False
    
    try:
        # prepare propagation
        c = input_data.c
        G = input_data.G
        GI = input_data.GI
        E = input_data.E
        id_ = input_data.id
        id_max = max(id_) if id_ else 0
        
        # make sure all properties have correct size
        if len(G) == 0:
            G = np.zeros((c.shape[0], 0))
            E = np.zeros((0, 0))
            id_ = []
            id_max = 1
        if len(GI) == 0:
            GI = np.zeros((c.shape[0], 0))
        if id_max == 0:
            id_max = 0
        
        # find all even exponents, also save others
        # (TL: this was done for speed, not sure how important it really is...)
        if len(E) > 0:
            ind = np.where(np.prod(np.ones_like(E) - (E % 2), axis=0) == 1)[0]
            ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        else:
            ind = np.array([], dtype=int)
            ind_ = np.array([], dtype=int)
        
        if options.get('nn', {}).get('order_reduction_sensitivity', False):
            # set sensitivity in each layer (used for order reduction)
            self.calcSensitivity(c)
        
        # iterate over all layers
        for k in idxLayer:
            if 'nn' not in options:
                options['nn'] = {}
            options['nn']['layer_k'] = k
            layer_k = self.layers[k]
            
            c, G, GI, E, id_, id_max, ind, ind_ = layer_k.evaluatePolyZonotope(
                c, G, GI, E, id_, id_max, ind, ind_, options)
            options = self._aux_updateOptions(options, 'polyZonotope', k, layer_k)
        
        # build result
        # This would require PolyZonotope constructor
        # For now, return a simplified result
        r = type(input_data)(c, G, GI, E, id_)
        
    except MemoryError:
        raise MemoryError("Out of memory while processing layer. Try to set options.nn.num_generators to a lower value.")
    
    if isZonotope:
        # transform back to zonotope
        # This would require Zonotope constructor
        r = type(input_data)(r.c, r.G)
    
    return r

def _aux_evaluateTaylm(self, input_data: Any, options: Dict[str, Any], idxLayer: List[int]) -> Any:
    """Evaluate Taylor model input"""
    r = input_data
    for k in idxLayer:
        if 'nn' not in options:
            options['nn'] = {}
        options['nn']['layer_k'] = k
        layer_k = self.layers[k]
        r = layer_k.evaluateTaylm(r, options)
        options = self._aux_updateOptions(options, 'taylm', k, layer_k)
    
    return r

def _aux_evaluateConZonotope(self, input_data: Any, options: Dict[str, Any], idxLayer: List[int]) -> Any:
    """Evaluate constrained zonotope input"""
    # convert constrained zonotope to star set
    # This would require nnHelper.conversionConZonoStarSet
    # For now, we'll use a simplified approach
    c = input_data.center
    G = input_data.generators
    C = input_data.C
    d = input_data.d
    l = input_data.l
    u = input_data.u
    
    for k in idxLayer:
        if 'nn' not in options:
            options['nn'] = {}
        options['nn']['layer_k'] = k
        layer_k = self.layers[k]
        c, G, C, d, l, u = layer_k.evaluateConZonotope(c, G, C, d, l, u, options)
        options = self._aux_updateOptions(options, 'conZonotope', k, layer_k)
    
    # convert star set back to constrained zonotope
    # This would require nnHelper.conversionStarSetConZono
    # For now, return a simplified result
    r = type(input_data)(c, G, C, d, l, u)
    
    return r

def _aux_updateOptions(self, options: Dict[str, Any], type_: str, k: int, layer_k: Any) -> Dict[str, Any]:
    """Update options during evaluation"""
    if type_ == 'polyZonotope':
        self.propagateBounds(k, options)
    
    # Check if it's a GNN projection layer
    if hasattr(layer_k, '__class__') and 'nnGNNProjectionLayer' in str(layer_k.__class__):
        # update graph
        if 'nn' in options and 'graph' in options['nn']:
            options['nn']['graph'] = options['nn']['graph'].subgraph(layer_k.nodes_keep)
            layer_k.updateMessagePassing()
    
    return options
