"""
priv_abstrerr_poly - computes the abstraction error for the polynomialization
   approach introduced in [1]

Syntax:
    [trueError, VerrorDyn, VerrorStat] = ...
           priv_abstrerr_poly(obj, options, Rall, Rdiff, H, Zdelta, ...
                               VerrorStat, T, ind3, Zdelta3)

Inputs:
    sys - nonlinearSys object
    Rall - time-interval reachable set
    Rdiff - difference between the reachable set at the beginning of the
            time interval and the time-interval reachable set
    params - model parameters
    options - options struct
    (...the following six input parameters come from priv_precompStatError.m)
    H - Hessian matrix
    Zdelta - zonotope over-approximating the reachable set at the
             beginning of the time step extended by the input set
    VerrorStat - set of static linearization errors
    T - third-order tensor
    ind3 - indices of non-zero entries in the third-order tensor
    Zdelta3 - set Zdelta reduced to the zonotope order for the evaluation
              of the third-order tensor

Outputs:
    trueError - interval overapproximating the overall linearization error 
    VerrorDyn - zonotope overapproximating the dynamic linearization error
    VerrorStat - zonotope overapproximating the static linearization error

References: 
  [1] M. Althoff
      "Reachability Analysis of Nonlinear Systems using
          Conservative Polynomialization and Non-Convex Sets"

Other m-files required: priv_precompStatError.m
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       21-August-2012
Last update:   18-March-2016
               25-July-2016 (intervalhull replaced by interval)
               22-January-2018 (NK, fixed error for the sets)
               08-February-2018 (NK, higher-order-tensors + clean-up)
               21-April-2020 (simplification)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Tuple, Any, Dict, Optional
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_abstrerr_poly(sys: Any, Rall: Any, Rdiff: Any, params: Dict[str, Any],
                       options: Dict[str, Any], H: Any, Zdelta: Zonotope,
                       VerrorStat: Any, T: Optional[Any], ind3: Optional[list],
                       Zdelta3: Optional[Zonotope]) -> Tuple[Interval, Zonotope, Any]:
    """
    Computes the abstraction error for the polynomialization approach
    
    Args:
        sys: nonlinearSys object
        Rall: time-interval reachable set
        Rdiff: difference between the reachable set at the beginning of the
               time interval and the time-interval reachable set
        params: model parameters (must contain 'U' key)
        options: options struct (must contain 'tensorOrder', 'reductionTechnique',
                'errorOrder', 'intermediateOrder', and optionally 'errorOrder3')
        H: Hessian matrix (from priv_precompStatError)
        Zdelta: zonotope over-approximating the reachable set at the beginning
                of the time step extended by the input set (from priv_precompStatError)
        VerrorStat: set of static linearization errors (from priv_precompStatError)
        T: third-order tensor (from priv_precompStatError, None if tensorOrder < 4)
        ind3: indices of non-zero entries in the third-order tensor (from priv_precompStatError)
        Zdelta3: set Zdelta reduced to the zonotope order for the evaluation
                 of the third-order tensor (from priv_precompStatError)
        
    Returns:
        trueError: interval overapproximating the overall linearization error
        VerrorDyn: zonotope overapproximating the dynamic linearization error
        VerrorStat: zonotope overapproximating the static linearization error (updated)
    """
    
    # compute interval of reachable set
    # MATLAB: dx = interval(Rall);
    dx = Rall.interval()
    
    # MATLAB: totalInt_x = dx + sys.linError.p.x;
    totalInt_x = dx + sys.linError.p.x
    
    # compute intervals of input
    # MATLAB: du = interval(params.U);
    du = params['U'].interval()
    
    # MATLAB: totalInt_u = du + sys.linError.p.u;
    totalInt_u = du + sys.linError.p.u
    
    # obtain intervals and combined interval z
    # MATLAB: dz = [dx; du];
    dz = Interval.vertcat(dx, du)
    
    # compute zonotope of state and input
    # MATLAB: Rred_diff = reduce(zonotope(Rdiff),options.reductionTechnique,options.errorOrder);
    Rdiff_zono = Zonotope(Rdiff)
    Rred_diff = Rdiff_zono.reduce(options['reductionTechnique'], options['errorOrder'])
    
    # MATLAB: Z_diff = cartProd(Rred_diff,params.U);
    Z_diff = Rred_diff.cartProd_(params['U'])
    
    # second-order error
    # MATLAB: error_secondOrder_dyn = 0.5*(quadMap(Zdelta,Z_diff,H) ...
    #                                  + quadMap(Z_diff,Zdelta,H) + quadMap(Z_diff,H));
    # quadMap can be called as quadMap(Z1, Z2, H) for mixed case or quadMap(Z, H) for single case
    from cora_python.contSet.zonotope.quadMap import quadMap
    error_secondOrder_dyn = 0.5 * (quadMap(Zdelta, Z_diff, H) +
                                   quadMap(Z_diff, Zdelta, H) +
                                   quadMap(Z_diff, H))
    
    # third-order error
    # MATLAB: if options.tensorOrder == 3
    if options['tensorOrder'] == 3:
        
        # set handles to correct files
        # MATLAB: sys = setHessian(sys,'standard');
        sys = sys.setHessian('standard')
        
        # MATLAB: sys = setThirdOrderTensor(sys,'int');
        sys = sys.setThirdOrderTensor('int')
        
        # evaluate the third-order tensor
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
            
            # evaluate third order tensor 
            # MATLAB: if isa(sys,'nonlinParamSys')
            is_nonlinParamSys = hasattr(sys, '__class__') and 'nonlinParamSys' in sys.__class__.__name__.lower()
            
            if is_nonlinParamSys and 'paramInt' in params:
                # MATLAB: [T,ind] = sys.thirdOrderTensor(objX, objU, params.paramInt);
                T, ind = sys.thirdOrderTensor(objX, objU, params['paramInt'])
            else:
                # MATLAB: [T,ind] = sys.thirdOrderTensor(objX, objU);
                T, ind = sys.thirdOrderTensor(objX, objU)
        
        else:
            # MATLAB: if isa(sys,'nonlinParamSys')
            is_nonlinParamSys = hasattr(sys, '__class__') and 'nonlinParamSys' in sys.__class__.__name__.lower()
            
            if is_nonlinParamSys and 'paramInt' in params:
                # MATLAB: [T,ind] = sys.thirdOrderTensor(totalInt_x, totalInt_u,params.paramInt);
                T, ind = sys.thirdOrderTensor(totalInt_x, totalInt_u, params['paramInt'])
            else:
                # MATLAB: [T,ind] = sys.thirdOrderTensor(totalInt_x,totalInt_u);
                T, ind = sys.thirdOrderTensor(totalInt_x, totalInt_u)
        
        # calculate the Lagrange remainder term
        # MATLAB: error_thirdOrder_dyn = interval(zeros(sys.nrOfDims,1),zeros(sys.nrOfDims,1));
        error_thirdOrder_dyn = Interval(np.zeros(sys.nrOfDims), np.zeros(sys.nrOfDims))
        
        # MATLAB: for i=1:length(ind)
        for i in range(len(ind)):
            # MATLAB: error_sum = interval(0,0);
            error_sum = Interval(0, 0)
            
            # MATLAB: for j=1:length(ind{i})
            for j in range(len(ind[i])):
                # MATLAB: error_sum = error_sum + (dz.'*T{i,ind{i}(j)}*dz) * dz(ind{i}(j));
                # Compute dz' * T{i,ind{i}(j)} * dz using proper interval arithmetic
                T_ij = T[i][ind[i][j]]
                dz_T = dz.transpose()  # Row vector (1 x n)
                dz_T_Tij = dz_T.mtimes(T_ij)  # (1 x n) interval
                error_tmp = dz_T_Tij.mtimes(dz)  # Scalar interval
                
                # dz(ind{i}(j)) is the ind{i}(j)-th element of dz interval
                dz_idx = Interval(dz.inf[ind[i][j]], dz.sup[ind[i][j]])
                error_sum = error_sum + error_tmp * dz_idx
            
            # MATLAB: error_thirdOrder_dyn(i,1) = 1/6*error_sum;
            error_thirdOrder_dyn.inf[i] = (1.0 / 6.0) * error_sum.inf
            error_thirdOrder_dyn.sup[i] = (1.0 / 6.0) * error_sum.sup
        
        # MATLAB: error_thirdOrder_dyn = zonotope(error_thirdOrder_dyn);
        error_thirdOrder_dyn = error_thirdOrder_dyn.zonotope()
        
        # no terms of order >= 4
        # MATLAB: remainder = zonotope(zeros(sys.nrOfDims,1));
        remainder = Zonotope(np.zeros((sys.nrOfDims, 1)))
    
    else:
        # tensorOrder >= 4
        # MATLAB: else
        
        # set handles to correct files
        # MATLAB: sys = setHessian(sys,'standard');
        sys = sys.setHessian('standard')
        
        # MATLAB: sys = setThirdOrderTensor(sys,'standard');
        sys = sys.setThirdOrderTensor('standard')
        
        # reduce set Zdiff to the desired zonotope order to speed up the
        # computation of cubic multiplication
        # MATLAB: if isfield(options,'errorOrder3')
        if 'errorOrder3' in options:
            # MATLAB: Rred_diff = reduce(Rred_diff,options.reductionTechnique,options.errorOrder3);
            Rred_diff = Rred_diff.reduce(options['reductionTechnique'], options['errorOrder3'])
            
            # MATLAB: Z_diff3 = cartProd(Rred_diff,params.U);
            Z_diff3 = Rred_diff.cartProd_(params['U'])
        else:
            # MATLAB: Z_diff3 = Z_diff;
            Z_diff3 = Z_diff
        
        # third-order error
        # MATLAB: error_thirdOrder_dyn = 1/6*(cubMap(Zdelta3,T,ind3) + ...
        #                                cubMap(Zdelta3,Zdelta3,Z_diff3,T,ind3) + ...
        #                                cubMap(Zdelta3,Z_diff3,Z_diff3,T,ind3) + ...
        #                                cubMap(Zdelta3,Z_diff3,Zdelta3,T,ind3) + ... 
        #                                cubMap(Z_diff3,Zdelta3,Z_diff3,T,ind3) + ...
        #                                cubMap(Z_diff3,Zdelta3,Zdelta3,T,ind3) + ...
        #                                cubMap(Z_diff3,Z_diff3,Zdelta3,T,ind3));
        # cubMap can be called as cubMap(Z, T, ind) for single case or cubMap(Z1, Z2, Z3, T, ind) for mixed case
        from cora_python.contSet.zonotope.cubMap import cubMap
        error_thirdOrder_dyn = (1.0 / 6.0) * (
            cubMap(Zdelta3, T, ind3) +
            cubMap(Zdelta3, Zdelta3, Z_diff3, T, ind3) +
            cubMap(Zdelta3, Z_diff3, Z_diff3, T, ind3) +
            cubMap(Zdelta3, Z_diff3, Zdelta3, T, ind3) +
            cubMap(Z_diff3, Zdelta3, Z_diff3, T, ind3) +
            cubMap(Z_diff3, Zdelta3, Zdelta3, T, ind3) +
            cubMap(Z_diff3, Z_diff3, Zdelta3, T, ind3)
        )
        
        # init higher-order error
        # MATLAB: remainder = interval(zeros(sys.nrOfDims,1),zeros(sys.nrOfDims,1));
        remainder = Interval(np.zeros(sys.nrOfDims), np.zeros(sys.nrOfDims))
        
        # exact evaluation of intermediate taylor terms
        # MATLAB: for i=4:options.tensorOrder-1
        for i in range(4, options['tensorOrder']):
            # MATLAB: handle = sys.tensors{i-3};
            handle = sys.tensors[i - 3]
            
            # MATLAB: remainder = remainder + handle(sys.linError.p.x,sys.linError.p.u,dx,du);
            remainder = remainder + handle(sys.linError.p.x, sys.linError.p.u, dx, du)
        
        # lagrange remainder over-approximating the last taylor term
        # MATLAB: handle = sys.tensors{options.tensorOrder-3};
        handle = sys.tensors[options['tensorOrder'] - 3]
        
        # MATLAB: if isa(sys,'nonlinParamSys')
        is_nonlinParamSys = hasattr(sys, '__class__') and 'nonlinParamSys' in sys.__class__.__name__.lower()
        
        if is_nonlinParamSys and 'paramInt' in params:
            # MATLAB: remainder = remainder + handle(totalInt_x,totalInt_u,dx,du,params.paramInt);
            remainder = remainder + handle(totalInt_x, totalInt_u, dx, du, params['paramInt'])
        else:
            # MATLAB: remainder = remainder + handle(totalInt_x,totalInt_u,dx,du);
            remainder = remainder + handle(totalInt_x, totalInt_u, dx, du)
        
        # MATLAB: remainder = zonotope(remainder);
        remainder = remainder.zonotope()
    
    # combine results
    # MATLAB: VerrorDyn = error_secondOrder_dyn + error_thirdOrder_dyn + remainder;
    VerrorDyn = error_secondOrder_dyn + error_thirdOrder_dyn + remainder
    
    # MATLAB: VerrorDyn = reduce(VerrorDyn,options.reductionTechnique,options.intermediateOrder);
    VerrorDyn = VerrorDyn.reduce(options['reductionTechnique'], options['intermediateOrder'])
    
    # MATLAB: errorIHabs = abs(interval(VerrorDyn) + interval(VerrorStat));
    VerrorDyn_interval = VerrorDyn.interval()
    VerrorStat_interval = VerrorStat.interval()
    errorIHabs = (VerrorDyn_interval + VerrorStat_interval).abs()
    
    # MATLAB: trueError = supremum(errorIHabs);
    trueError = errorIHabs.supremum()
    
    return trueError, VerrorDyn, VerrorStat

