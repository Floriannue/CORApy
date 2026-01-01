"""
outputSet - calculates output set based on output equation given by
   y = Cx + Du + k + Fv and sets for x (R) and u (params.U + params.uTrans)

Syntax:
    Y = outputSet(linsys,R,params,options)

Inputs:
    linsys - linearSys object
    R - reachable set (either time point [i] or time interval [i,i+1])
    params - model parameters
    options - options for the computation of reachable sets

Outputs:
    Y - output set (either time point [i] or time interval [i,i+1])

Authors:       Mark Wetzlinger
Written:       12-August-2019
Last update:   20-August-2019
               16-November-2021 (MW, add sensor noise V)
               19-November-2021 (MW, shift index of time-point solution)
               19-November-2022 (MW, remove double computation)
               07-December-2022 (MW, allow to skip output set)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def outputSet(linsys: Any, R: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Calculates output set based on output equation given by
    y = Cx + Du + k + Fv
    
    Args:
        linsys: linearSys object
        R: reachable set (either time point [i] or time interval [i,i+1])
        params: model parameters
        options: options for the computation of reachable sets
        
    Returns:
        Y: output set (either time point [i] or time interval [i,i+1])
    """
    
    # skip computation of output set
    if not options.get('compOutputSet', True):
        return R
    
    # output equation is not provided or y = x
    if (linsys.C is None or linsys.C.size == 0 or
        (np.isscalar(linsys.C) and linsys.C == 1 and
         not np.any(linsys.D) and
         not np.any(linsys.k) and
         (linsys.F is None or linsys.F.size == 0 or
          (hasattr(params.get('V', None), 'representsa_') and
           params.get('V', None).representsa_('origin', np.finfo(float).eps)) if params.get('V', None) is not None else True))):
        return R
    
    # do we consider inputs in the output?
    isD = False
    if linsys.D is not None and linsys.D.size > 0 and np.any(linsys.D):
        isD = True
        U = params.get('U', Zonotope(np.zeros((linsys.B.shape[1], 1)), np.array([]).reshape(linsys.B.shape[1], 0))) + params.get('uTrans', np.zeros((linsys.B.shape[1], 1)))
    
    if 'saveOrder' not in options:
        # output equations without order reduction
        if isD:
            Y = linsys.C @ R + linsys.D @ U + linsys.k + linsys.F @ params.get('V', Zonotope(np.zeros((linsys.F.shape[1], 1)), np.array([]).reshape(linsys.F.shape[1], 0))) if linsys.F is not None and linsys.F.size > 0 else Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0)))
        else:
            Y = linsys.C @ R + linsys.k + (linsys.F @ params.get('V', Zonotope(np.zeros((linsys.F.shape[1], 1)), np.array([]).reshape(linsys.F.shape[1], 0))) if linsys.F is not None and linsys.F.size > 0 else Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0)))
    else:
        # reduction by saveOrder
        if isD:
            Y = Zonotope(linsys.C @ R).reduce(options.get('reductionTechnique', 'girard'), options['saveOrder']) + linsys.D @ U + linsys.k + (linsys.F @ params.get('V', Zonotope(np.zeros((linsys.F.shape[1], 1)), np.array([]).reshape(linsys.F.shape[1], 0))) if linsys.F is not None and linsys.F.size > 0 else Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0)))
        else:
            Y = Zonotope(linsys.C @ R).reduce(options.get('reductionTechnique', 'girard'), options['saveOrder']) + linsys.k + (linsys.F @ params.get('V', Zonotope(np.zeros((linsys.F.shape[1], 1)), np.array([]).reshape(linsys.F.shape[1], 0))) if linsys.F is not None and linsys.F.size > 0 else Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0)))
    
    return Y

