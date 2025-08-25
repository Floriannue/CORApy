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
from ..nnHelper import validateNNoptions


def evaluate(obj: NeuralNetwork, input_data: Any, options: Optional[Dict[str, Any]] = None, 
            idxLayer: Optional[List[int]] = None) -> Any:
    """
    Compute the output of a neural network for the given input
    
    Args:
        obj: NeuralNetwork object
        input_data: input represented as a numeric or set
        options: options for neural network evaluation
        idxLayer: indices of layers that should be evaluated (defaults to all layers)
        
    Returns:
        res: output of the neural network
    """
    # parse input
    if options is None:
        options = {}
    if idxLayer is None:
        # MATLAB: 1:length(obj.layers)
        idxLayer = list(range(len(obj.layers)))  # 0-based indexing like Python
    
    # validate input
    # Note: Python doesn't have MATLAB's inputArgsCheck, so we'll implement basic validation
    if not hasattr(obj, '__class__') or 'neuralNetwork' not in str(obj.__class__):
        raise ValueError("First argument must be a neuralNetwork object")
    
    # validate input types (basic check)
    valid_input_types = ['numeric', 'interval', 'zonotope', 'polyZonotope', 
                        'taylm', 'conZonotope', 'gpuArray']
    # For now, we'll accept any input and let the evaluate_ method handle validation
    
    if not isinstance(options, dict):
        raise ValueError("Options must be a dictionary")
    
    if not isinstance(idxLayer, (list, tuple)) or not all(isinstance(i, int) for i in idxLayer):
        raise ValueError("idxLayer must be a numeric vector")
    
    options = validateNNoptions(options)
    
    # evaluate ----------------------------------------------------------------
    r = obj.evaluate_(input_data, options, idxLayer)
    
    return r
