"""
priv_abstrerr_lin - computes the abstraction error for linearization approach;
   to enter, options.alg = 'lin' and options.tensorOrder = 2|3

Syntax:
    [trueError,VerrorDyn] = priv_abstrerr_lin(sys,R,params,options)

Inputs:
    sys - nonlinearSys or nonlinParamSys object
    R - reachable set (time-interval solution from linearized system
           + estimated set of abstraction errors)
    params - model parameters
    options - options struct

Outputs:
    trueError - abstraction error (interval)
    Verrordyn - abstraction error (zonotope)

References: 
  [1] M. Althoff et al. "Reachability Analysis of Nonlinear Systems with 
      Uncertain Parameters using Conservative Linearization"

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: linReach

Former files: contDynamivs/priv_linError.m

Authors:       Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
Written:       21-April-2020
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Tuple, Any, Dict, List, Optional
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_abstrerr_lin(sys: Any, R: Any, params: Dict[str, Any], 
                      options: Dict[str, Any]) -> Tuple[np.ndarray, Zonotope]:
    """
    Computes the abstraction error for linearization approach
    
    Args:
        sys: nonlinearSys or nonlinParamSys object
        R: reachable set (time-interval solution from linearized system + estimated set of abstraction errors)
        params: model parameters (must contain 'U' key)
        options: options struct (must contain 'tensorOrder', 'reductionTechnique', 'errorOrder', 'intermediateOrder')
        
    Returns:
        trueError: abstraction error (numpy array, interval bounds)
        VerrorDyn: abstraction error (zonotope)
    """
    
    # compute interval of reachable set
    # MATLAB: IHx = interval(R);
    IHx = R.interval()
    
    # compute intervals of total reachable set
    # MATLAB: totalInt_x = IHx + sys.linError.p.x;
    totalInt_x = IHx + sys.linError.p.x
    
    # compute intervals of input
    # MATLAB: IHu = interval(params.U);
    IHu = params['U'].interval()
    
    # translate intervals by linearization point
    # MATLAB: totalInt_u = IHu + sys.linError.p.u;
    totalInt_u = IHu + sys.linError.p.u
    
    # MATLAB: if options.tensorOrder == 2
    if options['tensorOrder'] == 2:
        
        # assign correct hessian (using interval arithmetic)
        # MATLAB: sys = setHessian(sys,'int');
        sys = sys.setHessian('int')
        
        # evaluate the hessian matrix with the selected range-bounding technique
        # MATLAB: if isfield(options,'lagrangeRem') && isfield(options.lagrangeRem,'method') && ...
        #        ~strcmp(options.lagrangeRem.method,'interval')
        use_range_bounding = ('lagrangeRem' in options and 
                             'method' in options['lagrangeRem'] and
                             options['lagrangeRem']['method'] != 'interval')
        
        if use_range_bounding:
            # create taylor models or zoo-objects
            # MATLAB: [objX,objU] = initRangeBoundingObjects(totalInt_x,totalInt_u,options);
            from cora_python.g.functions.helper.sets.contSet.taylm.initRangeBoundingObjects import initRangeBoundingObjects
            objX, objU = initRangeBoundingObjects(totalInt_x, totalInt_u, options)
            
            # evaluate the Lagrange remainder 
            # MATLAB: if isa(sys,'nonlinParamSys')
            is_nonlinParamSys = hasattr(sys, '__class__') and 'nonlinParamSys' in sys.__class__.__name__.lower()
            
            if is_nonlinParamSys and 'paramInt' in params:
                # MATLAB: H = sys.hessian(objX,objU,params.paramInt);
                H = sys.hessian(objX, objU, params['paramInt'])
            else:
                # MATLAB: H = sys.hessian(objX,objU);
                H = sys.hessian(objX, objU)
        else:
            # MATLAB: if isa(sys,'nonlinParamSys')
            is_nonlinParamSys = hasattr(sys, '__class__') and 'nonlinParamSys' in sys.__class__.__name__.lower()
            
            if is_nonlinParamSys and 'paramInt' in params:
                # MATLAB: H = sys.hessian(totalInt_x,totalInt_u,params.paramInt);
                H = sys.hessian(totalInt_x, totalInt_u, params['paramInt'])
            else:
                # MATLAB: H = sys.hessian(totalInt_x,totalInt_u);
                H = sys.hessian(totalInt_x, totalInt_u)
        
        # MATLAB: if true
        # Standard method using interval arithmetic
        if True:
            
            # obtain maximum absolute values within IHx, IHu
            # MATLAB: dx = max(abs(infimum(IHx)),abs(supremum(IHx)));
            # MATLAB: du = max(abs(infimum(IHu)),abs(supremum(IHu)));
            dx = np.maximum(np.abs(IHx.infimum()), np.abs(IHx.supremum()))
            du = np.maximum(np.abs(IHu.infimum()), np.abs(IHu.supremum()))
            
            # Ensure column vectors (MATLAB always returns column vectors)
            # MATLAB: dx and du are column vectors
            if dx.ndim == 1:
                dx = dx.reshape(-1, 1)
            elif dx.shape[1] != 1:
                dx = dx.reshape(-1, 1)
            if du.ndim == 1:
                du = du.reshape(-1, 1)
            elif du.shape[1] != 1:
                du = du.reshape(-1, 1)
        
            # calculate the Lagrange remainder (second-order error)
            # ...acc. to Proposition 1 in [1]
            # MATLAB: errorLagr = zeros(length(H),1);
            errorLagr = np.zeros(len(H))
            
            # MATLAB: dz = [dx;du];
            dz = np.vstack([dx, du])
        
            # MATLAB: for i = 1:length(H)
            for i in range(len(H)):
                # MATLAB: H_ = abs(H{i});
                # H[i] is an Interval object (interval matrix)
                # In MATLAB, abs() on interval returns interval with absolute bounds
                # Then max(infimum, supremum) gets element-wise maximum
                if isinstance(H[i], Interval):
                    # For interval, use Python's built-in abs() (not np.abs)
                    # abs() on interval returns interval with absolute bounds
                    H_abs = abs(H[i])  # Uses Interval.__abs__
                    # Then get max of inf and sup element-wise
                    H_ = np.maximum(H_abs.infimum(), H_abs.supremum())
                else:
                    H_ = np.abs(H[i])
                    # If result is still interval, get max of inf and sup
                    if isinstance(H_, Interval):
                        H_abs = abs(H_)
                        H_ = np.maximum(H_abs.infimum(), H_abs.supremum())
                
                # Handle object arrays that might contain intervals
                if isinstance(H_, np.ndarray):
                    # If H_ is already a numpy array, check if elements are intervals
                    if H_.dtype == object:
                        # Object array - might contain intervals
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
                
                # MATLAB: errorLagr(i) = 0.5 * dz' * H_ * dz;
                errorLagr[i] = 0.5 * dz.T @ H_ @ dz
        
            # check if Lagrange remainder is too large
            # MATLAB: if any(isnan(errorLagr)) || any(isinf(errorLagr))
            if np.any(np.isnan(errorLagr)) or np.any(np.isinf(errorLagr)):
                raise CORAerror('CORA:reachSetExplosion', 'Lagrange remainder exploded.')
            
            # MATLAB: trueError = errorLagr;
            trueError = errorLagr
            
            # MATLAB: VerrorDyn = zonotope(zeros(size(trueError)),diag(trueError));
            VerrorDyn = Zonotope(np.zeros_like(trueError), np.diag(trueError))
        
        else:
            # no interval arithmetic (commented out in MATLAB)
            # This branch is not executed (if true is always True)
            pass
    
    # MATLAB: elseif options.tensorOrder == 3
    elif options['tensorOrder'] == 3:
        
        # set handles to correct files
        # MATLAB: sys = setHessian(sys,'standard');
        sys = sys.setHessian('standard')
        
        # MATLAB: sys = setThirdOrderTensor(sys,'int');
        sys = sys.setThirdOrderTensor('int')
        
        # obtain intervals and combined interval z
        # MATLAB: dz = [IHx; IHu];
        # Concatenate intervals vertically using vertcat
        dz = Interval.vertcat(IHx, IHu)
        
        # reduce zonotope
        # MATLAB: Rred = reduce(R,options.reductionTechnique,options.errorOrder);
        Rred = R.reduce(options['reductionTechnique'], options['errorOrder'])
        
        # combined zonotope (states + input)
        # MATLAB: Z = cartProd(Rred,params.U);
        Z = Rred.cartProd_(params['U'])
        
        # calculate hessian matrix
        # MATLAB: if isa(sys,'nonlinParamSys')
        is_nonlinParamSys = hasattr(sys, '__class__') and 'nonlinParamSys' in sys.__class__.__name__.lower()
        
        if is_nonlinParamSys and 'paramInt' in params:
            # MATLAB: H = sys.hessian(sys.linError.p.x,sys.linError.p.u,params.paramInt);
            H = sys.hessian(sys.linError.p.x, sys.linError.p.u, params['paramInt'])
        else:
            # MATLAB: H = sys.hessian(sys.linError.p.x,sys.linError.p.u);
            H = sys.hessian(sys.linError.p.x, sys.linError.p.u)
        
        # evaluate third-order tensor 
        # MATLAB: if isfield(options,'lagrangeRem') && isfield(options.lagrangeRem,'method') && ...
        #        ~strcmp(options.lagrangeRem.method,'interval')
        use_range_bounding = ('lagrangeRem' in options and 
                             'method' in options['lagrangeRem'] and
                             options['lagrangeRem']['method'] != 'interval')
        
        if use_range_bounding:
            # create taylor models or zoo objects
            # MATLAB: [objX,objU] = initRangeBoundingObjects(totalInt_x,totalInt_u,options);
            from cora_python.g.functions.helper.sets.contSet.taylm.initRangeBoundingObjects import initRangeBoundingObjects
            objX, objU = initRangeBoundingObjects(totalInt_x, totalInt_u, options)
            
            # MATLAB: if isa(sys,'nonlinParamSys')
            if is_nonlinParamSys and 'paramInt' in params:
                # MATLAB: [T,ind] = sys.thirdOrderTensor(objX, objU, params.paramInt);
                T, ind = sys.thirdOrderTensor(objX, objU, params['paramInt'])
            else:
                # MATLAB: [T,ind] = sys.thirdOrderTensor(objX, objU);
                T, ind = sys.thirdOrderTensor(objX, objU)
        
        else:
            # MATLAB: if isa(sys,'nonlinParamSys')
            if is_nonlinParamSys and 'paramInt' in params:
                # MATLAB: [T,ind] = sys.thirdOrderTensor(totalInt_x, totalInt_u, params.paramInt);
                T, ind = sys.thirdOrderTensor(totalInt_x, totalInt_u, params['paramInt'])
            else:
                # MATLAB: [T,ind] = sys.thirdOrderTensor(totalInt_x, totalInt_u);
                T, ind = sys.thirdOrderTensor(totalInt_x, totalInt_u)
        
        # second-order error
        # MATLAB: errorSec = 0.5 * quadMap(Z,H);
        errorSec = 0.5 * Z.quadMap(H)
        
        # calculate the Lagrange remainder term (third-order error)
        # MATLAB: if isempty(cell2mat(ind))
        # cell2mat(ind) converts cell array to matrix - in Python, flatten list of lists
        ind_flat = []
        for sublist in ind:
            if isinstance(sublist, (list, np.ndarray)):
                ind_flat.extend(sublist)
            else:
                ind_flat.append(sublist)
        
        if len(ind_flat) == 0:
            # empty zonotope if all entries in T are empty
            # MATLAB: errorLagr = zonotope(zeros(sys.nrOfDims,1));
            errorLagr = Zonotope(np.zeros((sys.nr_of_dims, 1)))
        else:
            # skip tensors with all-zero entries using ind from tensor creation
            # MATLAB: errorLagr = interval(zeros(sys.nrOfDims,1),zeros(sys.nrOfDims,1));
            errorLagr = Interval(np.zeros(sys.nr_of_dims), np.zeros(sys.nr_of_dims))
            
            # MATLAB: for i=1:length(ind)
            for i in range(len(ind)):
                # MATLAB: error_sum = interval(0,0);
                error_sum = Interval(0, 0)
                
                # MATLAB: for j=1:length(ind{i})
                for j in range(len(ind[i])):
                    # MATLAB: error_tmp = dz.'*T{i,j}*dz;
                    # dz is an interval column vector (n x 1)
                    # dz.' is the transpose (1 x n row vector)
                    # T[i][j] is a matrix (n x n)
                    # Result: dz.' * T[i][j] * dz is a scalar interval
                    T_ij = T[i][j]
                    
                    # Compute dz.' * T[i][j] first: (1 x n) interval * (n x n) matrix = (1 x n) interval
                    # Use proper transpose method
                    dz_T = dz.transpose()  # Row vector (1 x n)
                    dz_T_Tij = dz_T.mtimes(T_ij)  # (1 x n) interval
                    
                    # Then multiply by dz: (1 x n) interval * (n x 1) interval = scalar interval
                    error_tmp = dz_T_Tij.mtimes(dz)  # Scalar interval
                    
                    # MATLAB: error_sum = error_sum + error_tmp * dz(j);
                    # dz(j) is the j-th element of dz interval
                    dz_j = Interval(dz.inf[j], dz.sup[j])
                    error_sum = error_sum + error_tmp * dz_j
                
                # MATLAB: errorLagr(i,1) = 1/6*error_sum;
                errorLagr.inf[i] = (1.0 / 6.0) * error_sum.inf
                errorLagr.sup[i] = (1.0 / 6.0) * error_sum.sup
            
            # MATLAB: errorLagr = zonotope(errorLagr);
            errorLagr = errorLagr.zonotope()
        
        # overall linearization error
        # MATLAB: VerrorDyn = errorSec + errorLagr;
        VerrorDyn = errorSec + errorLagr
        
        # MATLAB: VerrorDyn = reduce(VerrorDyn,options.reductionTechnique,options.intermediateOrder);
        VerrorDyn = VerrorDyn.reduce(options['reductionTechnique'], options['intermediateOrder'])
        
        # MATLAB: trueError = supremum(abs(interval(VerrorDyn)));
        VerrorDyn_interval = VerrorDyn.interval()
        trueError = VerrorDyn_interval.supremum()
        trueError = np.abs(trueError)
    
    else:
        # MATLAB: throw(CORAerror('CORA:notSupported',...));
        raise CORAerror('CORA:notSupported',
                       "No abstraction error computation for chosen tensor order!")
    
    return trueError, VerrorDyn

