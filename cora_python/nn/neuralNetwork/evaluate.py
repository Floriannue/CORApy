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
from ..nnHelper.validateNNoptions import validateNNoptions
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck


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
    
    # validate parameters (matches MATLAB setDefaultValues exactly)
    [options, idxLayer] = setDefaultValues([{}, list(range(1, len(obj.layers) + 1))], list(args))
    
    # validate input (matches MATLAB inputArgsCheck exactly)
    inputArgsCheck([
        [obj, 'att', 'neuralNetwork'],
        [input_data, 'att', ['numeric', 'interval', 'zonotope', 'polyZonotope', 'taylm', 'conZonotope', 'gpuArray']],
        [options, 'att', 'struct'],
        [idxLayer, 'att', 'numeric', 'vector']
    ])
    
    # validate options (matches MATLAB exactly)
    options = validateNNoptions(options)
    
    # evaluate (matches MATLAB evaluate_ call exactly)
    r = obj.evaluate_(input_data, options, idxLayer)
    
    return r

