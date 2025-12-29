"""
contractForwardBackward - implementation of the forward-backward
                          contractor acc. to Sec. 4.2.4 in [1]

Syntax:
    res = contractForwardBackward(f,dom)

Inputs:
    f - function handle for the constraint f(x) = 0
    dom - initial domain (class: interval)

Outputs:
    res - contracted domain (class: interval)

Example: 
    f = @(x) x(1)^2 + x(2)^2 - 4;
    dom = interval([1;1],[3;3]);
   
    res = contract(f,dom,'forwardBackward');

References:
    [1] L. Jaulin et al. "Applied Interval Analysis", 2006

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contract

Authors:       Zhuoling Li, Niklas Kochdumper
Written:       04-November-2019 
Last update:   03-May-2024 (TL, bug fix given f is constant)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Callable, Any

from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.specification.syntaxTree import syntaxTree
from cora_python.specification.backpropagation import backpropagation


def contractForwardBackward(f: Callable, dom: Interval) -> Interval:
    """
    Implementation of the forward-backward contractor
    
    Args:
        f: function handle for the constraint f(x) = 0
        dom: initial domain (interval object)
        
    Returns:
        res: contracted domain (interval object)
    """
    
    # Compute syntax tree nodes for all variables
    vars_list = []
    for i in range(dom.dim()):
        # Access i-th dimension of interval
        dom_i = Interval(dom.inf[i], dom.sup[i])
        vars_list.append(syntaxTree(dom_i, i))
    
    # Forward iteration: compute syntax tree
    try:
        synTree = f(vars_list)
    except Exception as ME:
        # Check if function takes no parameters (constant function)
        import inspect
        sig = inspect.signature(f)
        if len(sig.parameters) == 0:
            # No parameters allowed; function is constant
            # This can happen if e.g. for a polynomial all coefficients
            # are 0 except for the y-intercept.
            const_val = f()
            synTree = syntaxTree(Interval(const_val, const_val), dom.dim())
        else:
            raise ME
    
    # Backward iteration
    res = dom
    
    # Handle synTree as single object or array
    if isinstance(synTree, list):
        synTree_list = synTree
    elif hasattr(synTree, '__len__') and not isinstance(synTree, str):
        synTree_list = list(synTree) if hasattr(synTree, '__iter__') else [synTree]
    else:
        synTree_list = [synTree]
    
    for i in range(len(synTree_list)):
        try:
            res = backpropagation(synTree_list[i], Interval(0, 0), res)
        except Exception as ex:
            if hasattr(ex, 'identifier') and ex.identifier == 'CORA:emptySet':
                return None  # MATLAB returns []
            else:
                raise ex
    
    return res

