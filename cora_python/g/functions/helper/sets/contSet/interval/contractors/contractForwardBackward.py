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
    # #region agent log
    import json
    with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
        log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"contractForwardBackward.py:before_f_call","message":"Before calling f(vars_list)","data":{"dom_inf":dom.inf.tolist(),"dom_sup":dom.sup.tolist(),"vars_list_len":len(vars_list),"f_type":type(f).__name__},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    try:
        synTree = f(vars_list)
        # #region agent log
        with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
            synTree_type = type(synTree).__name__
            synTree_info = None
            if isinstance(synTree, list):
                synTree_info = {"type":"list","len":len(synTree),"first_type":type(synTree[0]).__name__ if len(synTree) > 0 else None}
            elif hasattr(synTree, 'value'):
                # Get more details about the syntax tree
                value_info = None
                if hasattr(synTree.value, 'inf') and hasattr(synTree.value, 'sup'):
                    value_info = {"type":"Interval","inf":synTree.value.inf.tolist() if hasattr(synTree.value.inf, 'tolist') else str(synTree.value.inf),"sup":synTree.value.sup.tolist() if hasattr(synTree.value.sup, 'tolist') else str(synTree.value.sup)}
                else:
                    value_info = {"type":type(synTree.value).__name__,"value":str(synTree.value)}
                nodes_info = None
                if hasattr(synTree, 'nodes') and synTree.nodes is not None:
                    nodes_info = {"len":len(synTree.nodes),"node_types":[type(n).__name__ if n is not None else None for n in synTree.nodes]}
                funHan_info = None
                if hasattr(synTree, 'funHan'):
                    funHan_info = {"has_funHan":synTree.funHan is not None,"funHan_type":type(synTree.funHan).__name__ if synTree.funHan is not None else None}
                synTree_info = {"type":"SyntaxTree","has_value":True,"operator":getattr(synTree, 'operator', None),"value_info":value_info,"nodes_info":nodes_info,"funHan_info":funHan_info}
            log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"contractForwardBackward.py:after_f_call","message":"After calling f(vars_list)","data":{"synTree_type":synTree_type,"synTree_info":synTree_info},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
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
            # #region agent log
            with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"contractForwardBackward.py:before_backprop","message":"Before backpropagation","data":{"i":i,"res_before_inf":res.inf.tolist(),"res_before_sup":res.sup.tolist()},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            res = backpropagation(synTree_list[i], Interval(0, 0), res)
            # #region agent log
            with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"contractForwardBackward.py:after_backprop","message":"After backpropagation","data":{"i":i,"res_after_inf":res.inf.tolist(),"res_after_sup":res.sup.tolist()},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
        except Exception as ex:
            # MATLAB catches both CORA:emptySet and potentially other errors
            # Check if it's an empty set or out of domain error (which also indicates empty)
            if hasattr(ex, 'identifier'):
                if ex.identifier == 'CORA:emptySet' or ex.identifier == 'CORA:outOfDomain':
                    return None  # MATLAB returns []
            raise ex
    
    return res

