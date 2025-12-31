"""
initRangeBoundingObjects - creates taylm- or zoo-objects for the state
   and input variables

Syntax:
    [objX, objU] = initRangeBoundingObjects(intX, intU, options)

Inputs:
    intX - interval bounding the state variables
    intU - interval bounding the input variables
    options - struct containing algorithm settings

Outputs:
    objX - taylm-/zoo-object for the state variables
    objU - taylm-/zoo-object for the input variables

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: taylm, zoo

Authors:       Niklas Kochdumper
Written:       02-August-2018
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Tuple, Any, Dict
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def initRangeBoundingObjects(intX: Interval, intU: Interval, 
                             options: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Creates taylm- or zoo-objects for the state and input variables
    
    Args:
        intX: interval bounding the state variables
        intU: interval bounding the input variables
        options: struct containing algorithm settings (must contain 'lagrangeRem' key)
        
    Returns:
        objX: taylm-/zoo-object for the state variables
        objU: taylm-/zoo-object for the input variables
    """
    
    # parse options
    # MATLAB: maxOrder = [];
    maxOrder = None
    optMethod = None
    eps = None
    tolerance = None
    
    # MATLAB: if isfield(options.lagrangeRem,'maxOrder')
    if 'lagrangeRem' in options and 'maxOrder' in options['lagrangeRem']:
        maxOrder = options['lagrangeRem']['maxOrder']
    
    # MATLAB: if isfield(options.lagrangeRem,'optMethod')
    if 'lagrangeRem' in options and 'optMethod' in options['lagrangeRem']:
        optMethod = options['lagrangeRem']['optMethod']
    
    # MATLAB: if isfield(options.lagrangeRem,'tolerance')
    # NOTE: MATLAB code has a bug - it checks for 'tolerance' but assigns eps (line 43)
    #       and checks for 'eps' but assigns tolerance (line 46)
    #       We implement the correct behavior here
    if 'lagrangeRem' in options and 'tolerance' in options['lagrangeRem']:
        # MATLAB bug: eps = options.lagrangeRem.eps; (should be tolerance)
        # Correct implementation:
        tolerance = options['lagrangeRem']['tolerance']
    
    # MATLAB: if isfield(options.lagrangeRem,'eps')
    if 'lagrangeRem' in options and 'eps' in options['lagrangeRem']:
        # MATLAB bug: tolerance = options.lagrangeRem.tolerance; (should be eps)
        # Correct implementation:
        eps = options['lagrangeRem']['eps']
    
    # generate taylor model or zoo objects
    # MATLAB: if strcmp(options.lagrangeRem.method,'taylorModel')
    if options['lagrangeRem']['method'] == 'taylorModel':
        
        # MATLAB: objX = taylm(intX,maxOrder,aux_idxVars('x',1:length(intX)), ...
        #                     optMethod,eps,tolerance);
        from cora_python.contSet.taylm import Taylm
        idxX = aux_idxVars('x', list(range(1, intX.dim() + 1)))
        objX = Taylm(intX, maxOrder, idxX, optMethod, eps, tolerance)
        
        # MATLAB: objU = taylm(intU,maxOrder,aux_idxVars('u',1:length(intU)), ...
        #                     optMethod,eps,tolerance);
        idxU = aux_idxVars('u', list(range(1, intU.dim() + 1)))
        objU = Taylm(intU, maxOrder, idxU, optMethod, eps, tolerance)
    
    # MATLAB: elseif strcmp(options.lagrangeRem.method,'zoo')
    elif options['lagrangeRem']['method'] == 'zoo':
        
        # MATLAB: objX = zoo(intX,options.lagrangeRem.zooMethods, ...
        #                   aux_idxVars('x',1:length(intX)),maxOrder,eps,tolerance);
        from cora_python.contSet.zoo import Zoo
        idxX = aux_idxVars('x', list(range(1, intX.dim() + 1)))
        objX = Zoo(intX, options['lagrangeRem']['zooMethods'], idxX, maxOrder, eps, tolerance)
        
        # MATLAB: objU = zoo(intU,options.lagrangeRem.zooMethods, ...
        #                   aux_idxVars('u',1:length(intU)),maxOrder,eps,tolerance);
        idxU = aux_idxVars('u', list(range(1, intU.dim() + 1)))
        objU = Zoo(intU, options['lagrangeRem']['zooMethods'], idxU, maxOrder, eps, tolerance)
    
    else:
        # MATLAB: throw(CORAerror('CORA:wrongFieldValue','options.lagrangeRem.method',...
        #            {'taylorModel','zoo'}));
        raise CORAerror('CORA:wrongFieldValue', 'options.lagrangeRem.method',
                       ['taylorModel', 'zoo'])
    
    return objX, objU


# Auxiliary functions -----------------------------------------------------

def aux_idxVars(varName: str, idx: list) -> list:
    """
    Creates indexed variable names
    
    Args:
        varName: base variable name (e.g., 'x' or 'u')
        idx: list of indices (1-based, will be converted to strings)
        
    Returns:
        indexedVars: list of variable name strings
    """
    # MATLAB: idxLength = length(idx);
    idxLength = len(idx)
    
    # MATLAB: indexedVars = cell(idxLength,1);
    indexedVars = []
    
    # MATLAB: for i=1:idxLength
    for i in range(idxLength):
        # MATLAB: indexedVars{i} = [varName,num2str(idx(i))];
        indexedVars.append(varName + str(idx[i]))
    
    return indexedVars

