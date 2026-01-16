"""
lin_error2dAB - computes the uncertainty interval to be added to system
   matrix set caused by Lagrangian remainder of the linearization

Syntax:
    [dA,dB] = lin_error2dAB(R,U,hessian,p,varargin)

Inputs:
    R - reachable set
    U - input set
    hessian - hessian function handle
    p - linearization point (dict with 'x' and 'u' keys)
    varargin - optional additional arguments for hessian

Outputs:
    dA,dB - deviations for A,B caused by lagrange remainder

Example: 
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: 
   -

Authors:       Victor Gassmann
Written:       14-May-2019 
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Tuple, Any, Callable, Dict, Optional
from cora_python.contSet.interval import Interval


def lin_error2dAB(R: Any, U: Any, hessian: Callable, p: Dict[str, np.ndarray],
                  *varargin: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the uncertainty interval to be added to system matrix set 
    caused by Lagrangian remainder of the linearization
    
    Args:
        R: reachable set
        U: input set
        hessian: hessian function handle
        p: linearization point (dict with 'x' and 'u' keys)
        *varargin: optional additional arguments for hessian
        
    Returns:
        dA: deviations for A matrix (n x n)
        dB: deviations for B matrix (n x m)
    """
    
    # init
    # MATLAB: totalInt = interval(R);
    totalInt = R.interval()
    
    # MATLAB: inputInt = interval(U);
    inputInt = U.interval()
    
    # MATLAB: dim_x = size(totalInt,1);
    dim_x = totalInt.dim()
    
    # MATLAB: dim_u = size(inputInt,1);
    dim_u = inputInt.dim()
    
    # MATLAB: dxInt = interval(R+(-p.x));
    dxInt = (R + (-p['x'])).interval()
    
    # MATLAB: duInt = interval(U+(-p.u));
    duInt = (U + (-p['u'])).interval()
    
    # compute hessian
    # MATLAB: H = hessian(totalInt,inputInt,varargin{:});
    H = hessian(totalInt, inputInt, *varargin)
    
    # init derivative variables
    # MATLAB: dx = max(abs(dxInt.inf),abs(dxInt.sup));
    dx = np.maximum(np.abs(dxInt.inf), np.abs(dxInt.sup))
    
    # MATLAB: du = max(abs(duInt.inf),abs(duInt.sup));
    du = np.maximum(np.abs(duInt.inf), np.abs(duInt.sup))
    
    # MATLAB: dz = [dx;du];
    dz = np.vstack([dx, du])
    
    # MATLAB: dA = zeros(dim_x,dim_x);
    dA = np.zeros((dim_x, dim_x))
    
    # MATLAB: dB = zeros(dim_x,dim_u);
    dB = np.zeros((dim_x, dim_u))
    
    # compute dA, dB
    # MATLAB: for i = 1:length(H)
    for i in range(len(H)):
        # MATLAB: H_ = abs(H{i});
        H_ = abs(H[i])
        
        # MATLAB: H_inf = infimum(H_);
        # MATLAB: H_sup = supremum(H_);
        # MATLAB: H_ = max(H_inf,H_sup);
        # H_ might be an interval matrix, so we need to handle it
        if isinstance(H_, Interval):
            H_inf = H_.infimum()
            H_sup = H_.supremum()
            H_ = np.maximum(H_inf, H_sup)
        elif isinstance(H_, np.ndarray) and H_.dtype == object:
            # Object array might contain intervals
            H_flat = H_.flatten()
            H_max = np.zeros_like(H_flat, dtype=float)
            for j in range(len(H_flat)):
                if isinstance(H_flat[j], Interval):
                    H_max[j] = max(H_flat[j].infimum(), H_flat[j].supremum())
                else:
                    H_max[j] = float(H_flat[j])
            H_ = H_max.reshape(H_.shape)
        else:
            # Regular numeric array
            H_ = np.abs(H_)
        
        # MATLAB: M = 1/2*dz'*H_;
        M = 0.5 * dz.T @ H_
        
        # MATLAB: dA(i,:) = M(:,1:dim_x);
        dA[i, :] = M[:, :dim_x]
        
        # MATLAB: dB(i,:) = M(:,dim_x+1:end);
        dB[i, :] = M[:, dim_x:]
    
    return dA, dB

