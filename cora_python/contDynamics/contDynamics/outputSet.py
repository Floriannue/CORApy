"""
outputSet - computes output set based on a (non-)linear output equation

Syntax:
    Y = outputSet(sys,R,params,options)
    [Y, Verror] = outputSet(sys,R,params,options)

Inputs:
    sys - contDynamics object
    R - reachable set (either time point [i] or time interval [i,i+1])
    params - model parameters
    options - options for the computation of reachable sets

Outputs:
    Y - output set (either time point [i] or time interval [i,i+1])
    Verror - linearization error

Authors:       Mark Wetzlinger
Written:       19-November-2022
Last update:   07-December-2022 (MW, allow to skip output set)
               23-June-2023 (LL, consider inputs in first-order term)
               06-November-2023 (LL, add Verror as output)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, Tuple, Union, List
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def outputSet(sys: Any, R: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Union[Any, Tuple[Any, Any]]:
    """
    Computes output set based on a (non-)linear output equation
    
    Args:
        sys: contDynamics object
        R: reachable set (either time point [i] or time interval [i,i+1])
        params: model parameters
        options: options for the computation of reachable sets
        
    Returns:
        Y: output set (either time point [i] or time interval [i,i+1])
        Verror: (optional) linearization error
    """
    
    # skip computation of output set
    if not options.get('compOutputSet', True):
        return R
    
    # linParamSys or linProbSys
    if (hasattr(sys, '__class__') and 
        (sys.__class__.__name__ == 'linParamSys' or sys.__class__.__name__ == 'linProbSys')):
        # ...adapt this when these classes also support output functions
        return R
    
    # nonlinear systems: check tensor order for evaluation of output equation
    tensorOrderOutput = options.get('tensorOrderOutput', 2)
    if tensorOrderOutput not in [2, 3]:
        raise CORAerror('CORA:notSupported',
                       'Only tensor orders 2 and 3 supported for computation of output set.')
    
    # only nonlinear systems from here on...
    
    if not isinstance(R, (list, tuple)):
        Y, Verror = _aux_outputSet(sys, tensorOrderOutput, R, params)
        # Return both Y and Verror (caller can unpack if needed)
        return (Y, Verror)
    else:
        Y = [None] * len(R)
        Verror = None
        for i in range(len(R)):
            if isinstance(R[i], dict) or hasattr(R[i], '__dict__'):
                # time-point solution
                Y_i, Verror = _aux_outputSet(sys, tensorOrderOutput, R[i].get('set', R[i]) if isinstance(R[i], dict) else getattr(R[i], 'set', R[i]), params)
                if isinstance(R[i], dict):
                    Y[i] = {'set': Y_i}
                    if 'prev' in R[i]:
                        Y[i]['prev'] = R[i]['prev']
                    if 'parent' in R[i]:
                        Y[i]['parent'] = R[i]['parent']
                else:
                    Y[i] = type(R[i])(set=Y_i)
                    if hasattr(R[i], 'prev'):
                        Y[i].prev = R[i].prev
                    if hasattr(R[i], 'parent'):
                        Y[i].parent = R[i].parent
            else:
                # time-interval solution
                Y[i], Verror = _aux_outputSet(sys, tensorOrderOutput, R[i], params)
        return Y, Verror


# Auxiliary functions -----------------------------------------------------

def _aux_outputSet(sys: Any, tensorOrderOutput: int, R: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Output set computation for a given reachable set
    """
    
    # dimension of state and inputs
    n = sys.nr_of_dims if hasattr(sys, 'nr_of_dims') else sys.nrOfDims
    m = sys.nr_of_inputs if hasattr(sys, 'nr_of_inputs') else sys.nrOfInputs
    r = sys.nr_of_outputs if hasattr(sys, 'nr_of_outputs') else sys.nrOfOutputs
    
    # input set
    U = params.get('U', Zonotope(np.zeros((m, 1)), np.array([]).reshape(m, 0)))
    
    # expansion points
    p_x = R.center() if hasattr(R, 'center') else np.zeros((n, 1))
    p_u = U.center() if hasattr(U, 'center') else np.zeros((m, 1))
    p = np.concatenate([p_x, p_u])
    
    if hasattr(sys, '__class__') and sys.__class__.__name__ == 'nonlinParamSys':
        if 'paramInt' in params and isinstance(params['paramInt'], Interval):
            p_p = params['paramInt'].center() if hasattr(params['paramInt'], 'center') else params['paramInt']
        else:
            p_p = params.get('paramInt', 0)
    
    # interval enclosure of set
    I_x = Interval(R) if not isinstance(R, Interval) else R
    I_u = Interval(U) if not isinstance(U, Interval) else U
    I = I_x.cartProd_(I_u)
    
    # evaluate output function and Jacobian at expansion point
    D_lin = np.zeros((r, m))
    
    class_name = sys.__class__.__name__.lower() if hasattr(sys, '__class__') else ''
    if class_name in ['nonlinearsys', 'nonlinearsysdt']:
        zerothorder = sys.out_mFile(p_x, p_u)
        J, D_lin = sys.out_jacobian(p_x, p_u)
    elif class_name == 'nonlinparamsys':
        if 'paramInt' in params and isinstance(params['paramInt'], Interval):
            # ...copied from nonlinParamSys/linearize
            # constant
            f0_3D = sys.out_parametricDynamicFile(p_x, p_u)
            f0cell = [f0_3D[:, :, i] for i in range(sys.nr_of_param + 1)]
            # normalize cells
            for i in range(1, len(f0cell)):
                f0cell[0] = f0cell[0] + params['paramInt'][i-1].center() * f0cell[i] if hasattr(params['paramInt'][i-1], 'center') else params['paramInt'][i-1] * f0cell[i]
                f0cell[i] = params['paramInt'][i-1].rad() * f0cell[i] if hasattr(params['paramInt'][i-1], 'rad') else params['paramInt'][i-1] * f0cell[i]
            # create constant input zonotope
            f0Mat = np.zeros((f0cell[0].shape[0], len(f0cell)))
            for i in range(len(f0cell)):
                f0Mat[:, i] = f0cell[i].flatten() if f0cell[i].ndim > 1 else f0cell[i]
            zerothorder = Zonotope(f0Mat)
            
            # Jacobian
            J = sys.out_jacobian(p_x, p_u)
            # create matrix zonotopes, convert to interval matrix for Jacobian
            from cora_python.matrixSet.matZonotope.matZonotope import matZonotope
            from cora_python.matrixSet.intervalMatrix.intervalMatrix import intervalMatrix
            J_matZono = matZonotope(J[:, :, 0], J[:, :, 1:])
            J = intervalMatrix(J_matZono)
        else:  # params.paramInt is numeric
            zerothorder = sys.out_mFile(p_x, p_u, p_p)
            J = sys.out_jacobian_freeParam(p_x, p_u, p_p)
    else:
        raise CORAerror('CORA:notSupported', 
                       f'outputSet not implemented for {sys.__class__.__name__}')
    
    # first-order
    firstorder = J @ (R + (-p_x)) + D_lin @ (U + (-p_u))
    
    is_linear = False
    if hasattr(sys, 'out_isLinear'):
        val = sys.out_isLinear
        try:
            is_linear = bool(np.all(val))
        except Exception:
            is_linear = bool(val)
    if is_linear:
        # only affine map
        secondorder = Zonotope(np.zeros((J.shape[0], 1)), np.array([]).reshape(J.shape[0], 0))
        thirdorder = np.zeros((r, 1))
    
    elif tensorOrderOutput == 2:
        
        # assign correct hessian (using interval arithmetic)
        if hasattr(sys, 'setOutHessian'):
            sys = sys.setOutHessian('int')
        
        # evaluate Hessian using interval arithmetic
        if hasattr(sys, '__class__') and sys.__class__.__name__ == 'nonlinParamSys':
            H = sys.out_hessian(I_x, I_u, params.get('paramInt', 0))
        else:
            H = sys.out_hessian(I_x, I_u)
        
        # obtain maximum absolute values within I
        dz = np.maximum(np.abs(I.infimum() if hasattr(I, 'infimum') else I.inf),
                       np.abs(I.supremum() if hasattr(I, 'supremum') else I.sup))
        
        # calculate the second-order error
        secondorder = np.zeros((len(H), 1))
        for i in range(len(H)):
            H_ = np.abs(H[i])
            H_ = np.maximum(H_.infimum() if hasattr(H_, 'infimum') else H_.inf,
                           H_.supremum() if hasattr(H_, 'supremum') else H_.sup)
            secondorder[i] = 0.5 * dz.T @ H_ @ dz
        
        secondorder = Zonotope(np.zeros((r, 1)), np.diag(secondorder.flatten()))
        
        # no third-order computation
        thirdorder = np.zeros((r, 1))
    
    elif tensorOrderOutput == 3:
        
        # set handles to correct files
        if hasattr(sys, 'setOutHessian'):
            sys = sys.setOutHessian('standard')
        if hasattr(sys, 'setOutThirdOrderTensor'):
            sys = sys.setOutThirdOrderTensor('int')
        
        # evaluate Hessians at expansion point
        if hasattr(sys, '__class__') and sys.__class__.__name__ == 'nonlinParamSys':
            H = sys.out_hessian(p_x, p_u, p_p)
        else:
            H = sys.out_hessian(p_x, p_u)
        
        # Cartesian product of given set of states and inputs
        from cora_python.contSet.contSet.cartProd import cartProd
        Z = cartProd(R, I_u)
        
        # second-order
        from cora_python.contSet.zonotope.quadMap import quadMap
        secondorder = 0.5 * quadMap(Z + (-p), H)
        
        # evaluate third-order tensors over entire set
        T, ind = sys.out_thirdOrderTensor(I_x, I_u)
        
        # compute Lagrange remainder
        I_shifted = I + (-p)
        thirdorder = Interval(np.zeros((r, 1)), np.zeros((r, 1)))
        
        for i in range(r):
            for j in range(n + m):
                if not (hasattr(T[i][j], 'representsa_') and T[i][j].representsa_('emptySet', 1e-12)):
                    thirdorder[i] = thirdorder[i] + I_shifted[j] * I_shifted.T @ T[i][j] @ I_shifted
        
        # include factor and convert remainder to zonotope for quadMap below
        thirdorder = Zonotope((1/6) * thirdorder)
    
    # compute output set (ensure contSet handles numeric offsets)
    Verror = secondorder + thirdorder
    if isinstance(zerothorder, np.ndarray):
        Y = firstorder + zerothorder
    else:
        Y = zerothorder + firstorder
    Y = Y + Verror
    
    return Y, Verror

