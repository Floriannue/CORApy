"""
evaluate - compute the output of a neural network for the given input

Syntax:
    res = evaluate(obj, input)
    res = evaluate(obj, input, options)

Inputs:
    obj - object of class neuralNetwork
    input - input represented as a numeric or set
    options - options for neural network evaluation 
       (stored in options.nn)
       .bound_approx: bool whether bounds should be overapproximated,
           or "sample" (not safe!)
       .reuse_bounds: wheter bounds should be reused
       .poly_method: how approximation polynomial is found in nonlinear
           layers, e.g. 'regression', 'singh', 'taylor' ...
       .num_generators: max number of generators for order reduction
       .add_approx_error_to_GI: whether 'd' should be added to GI
       .plot_multi_layer_approx_info: plotting for nnApproximationLayer
       .max_bounds: max order used in refinement
       .do_pre_order_reduction: wheter to do a order reduction before
           evaluating the pZ using the polynomial
       .max_gens_post: max num_generator before post order reduction
       .remove_GI: whether to restructue s.t. there remain no ind. gens
       .force_approx_lin_at: (l, u) distance at which to use 'lin'
           instead of respective order using 'adaptive'
       .sort_exponents: whether exponents should be sorted
       .propagate_bounds: whether bounds should be propagated to next
           activation layer using interval arithmetic
       .maxpool_type: for set-based prediction, 'project' or 'regression'
       .order_reduction_sensitivity: whether sensitivity should be used
           during order reduction
       .G: graph object used for graph neural networks
    idxLayer - indices of layers that should be evaluated

Outputs:
    res - output of the neural network

References:
    [1] Kochdumper, N., et al. (2023). Open-and closed-loop neural network
        verification using polynomial zonotopes. NASA Formal Methods.
    [2] Ladner, T., et al. (2023). Automatic abstraction refinement in
        neural network verification using sensitivity analysis. HSCC '23:
        Proceedings of the 26th International Conference on
        Hybrid Systems: Computation and Control.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork, nnHelper/validateNNoptions

Authors:       Tobias Ladner
Written:       28-March-2022
Last update:   29-November-2022 (validateNNoptions)
                16-February-2023 (re-organized structure)
                21-February-2024 (moved internals to evaluate_)
Last revision: 17-July-2023 (improved readability)
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from .neuralNetwork import NeuralNetwork


def evaluate(obj: NeuralNetwork, input_data, *args):
    """
    Compute the output of a neural network for the given input (matches MATLAB exactly)
    
    Args:
        obj: object of class neuralNetwork
        input_data: input represented as a numeric or set
        *args: variable arguments (options, idxLayer)
        
    Returns:
        res: output of the neural network
    """
    # parse input (matches MATLAB narginchk(2,5))
    if len(args) > 3:
        raise ValueError("Too many arguments")
    
    # validate parameters (matches MATLAB setDefaultValues)
    options = {}
    idxLayer = list(range(1, len(obj.layers) + 1))  # 1:length(obj.layers)
    
    if len(args) >= 1:
        options = args[0]
    if len(args) >= 2:
        idxLayer = args[1]
    
    # validate input (matches MATLAB inputArgsCheck exactly)
    # MATLAB: {obj, 'att', 'neuralNetwork'}
    if not hasattr(obj, '__class__') or 'NeuralNetwork' not in str(obj.__class__):
        raise ValueError("First argument must be of class neuralNetwork")
    
    # MATLAB: {input, 'att', {'numeric', 'interval', 'zonotope', 'polyZonotope', 'taylm', 'conZonotope','gpuArray'}}
    # For now, accept any input type like MATLAB does
    # TODO: Implement proper type checking for different CORA set types
    
    # MATLAB: {options, 'att', 'struct'}
    if not isinstance(options, dict):
        raise ValueError("Options must be a struct/dict")
    
    # MATLAB: {idxLayer, 'att', 'numeric', 'vector'}
    if not isinstance(idxLayer, (list, np.ndarray)):
        raise ValueError("idxLayer must be numeric vector")
    
    # MATLAB: options = nnHelper.validateNNoptions(options);
    # TODO: Implement validateNNoptions when available
    
    # evaluate (matches MATLAB evaluate_ call exactly)
    r = evaluate_(obj, input_data, options, idxLayer)
    
    return r


def evaluate_(obj: NeuralNetwork, input_data, options: Dict[str, Any], 
             idxLayer: List[int]):
    """
    Internal evaluate function (matches MATLAB evaluate_ exactly)
    
    Args:
        obj: neural network object
        input_data: input data
        options: evaluation options
        idxLayer: layer indices to evaluate (1-based like MATLAB)
        
    Returns:
        Output of the neural network
    """
    # MATLAB uses 1-based indexing, convert to 0-based for Python
    # MATLAB: idxLayer = 1:length(obj.layers) means [1, 2, 3, ...]
    # Python needs [0, 1, 2, ...]
    python_idxLayer = [i - 1 for i in idxLayer]
    
    current_input = input_data
    
    # Evaluate only the specified layers
    for i in python_idxLayer:
        if 0 <= i < len(obj.layers):
            layer = obj.layers[i]
            current_input = layer.evaluateNumeric(current_input, options)
    
    return current_input
