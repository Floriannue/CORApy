"""
priv_precompStatError - precompute the second order static error along with 
   Hessian matrix

Syntax:
    [H,Zdelta,errorStat,T,ind3] = priv_precompStatError(sys,Rdelta,params,options)

Inputs:
    sys - nonlinear system object
    Rdelta - shifted reachable set at the beginning of the time step
    params - model parameters
    options - options struct

Outputs:
    H - hessian matrix
    Zdelta - zonotope over-approximating the reachable set at the
             beginning of the time step extended by the input set 
    errorStat - static linearization error
    T - third-order tensor
    ind3 - indices at which the third-order tensor is not zero
    Zdelta3 - set Zdelta reduced to the zonotope order for the evaluation
              of the third-order tensor

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Niklas Kochdumper
Written:       27-July-2018
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Tuple, Any, Optional, Dict, List, Union
from cora_python.contSet.zonotope import Zonotope


def priv_precompStatError(sys: Any, Rdelta: Any, params: Dict[str, Any], 
                          options: Dict[str, Any]) -> Tuple[Any, Zonotope, Any, 
                                                           Optional[Any], Optional[List], Optional[Zonotope]]:
    """
    Precompute the second order static error along with Hessian matrix
    
    Args:
        sys: nonlinear system object (nonlinearSys or nonlinParamSys)
        Rdelta: shifted reachable set at the beginning of the time step
        params: model parameters (must contain 'U' key)
        options: options struct (must contain 'reductionTechnique', 'errorOrder', 
                'tensorOrder', 'intermediateOrder', and optionally 'errorOrder3')
        
    Returns:
        H: Hessian matrix (list of matrices)
        Zdelta: zonotope over-approximating the reachable set at the beginning 
                of the time step extended by the input set
        errorStat: static linearization error (zonotope)
        T: third-order tensor (list of matrices, None if tensorOrder < 4)
        ind3: indices at which the third-order tensor is not zero (list, None if tensorOrder < 4)
        Zdelta3: set Zdelta reduced to the zonotope order for the evaluation
                 of the third-order tensor (None if tensorOrder < 4 or errorOrder3 not specified)
    """
    
    # set handle to correct file
    # MATLAB: sys = setHessian(sys,'standard');
    sys = sys.setHessian('standard')
    
    # initialize output arguments
    T = None
    ind3 = None
    Zdelta3 = None
    
    # reduce the reachable set for the initial time point
    # MATLAB: Rred = reduce(Rdelta,options.reductionTechnique,options.errorOrder);
    Rred = Rdelta.reduce(options['reductionTechnique'], options['errorOrder'])
    
    # compute a zonotope over-approximation of the reachable set at the
    # initial time point
    # MATLAB: if isa(Rdelta,'zonotope')
    if isinstance(Rdelta, Zonotope):
        Rdelta = Rred
    else:
        # MATLAB: Rdelta = reduce(zonotope(Rdelta),options.reductionTechnique,options.errorOrder);
        Rdelta_zono = Zonotope(Rdelta)
        Rdelta = Rdelta_zono.reduce(options['reductionTechnique'], options['errorOrder'])
    
    # extend the sets by the input sets
    # MATLAB: Ustat = zonotope(zeros(dim(params.U),1));
    # In Python, sets have .dim() method - use it directly
    if hasattr(params['U'], 'dim'):
        U_dim = params['U'].dim()
    else:
        # If U doesn't have dim method, it's not a valid set object
        raise AttributeError(f"params['U'] must be a set object with .dim() method, got {type(params['U'])}")
    Ustat = Zonotope(np.zeros((U_dim, 1)))
    
    # MATLAB: Z = cartProd(Rred,Ustat);
    Z = Rred.cartProd_(Ustat)
    
    # MATLAB: Zdelta = cartProd(Rdelta,Ustat);
    Zdelta = Rdelta.cartProd_(Ustat)
    
    # calculate the hessian tensor
    # MATLAB: if isa(sys,'nonlinParamSys')
    # Check if sys is nonlinParamSys by checking for paramInt in params or checking class name
    is_nonlinParamSys = hasattr(sys, '__class__') and 'nonlinParamSys' in sys.__class__.__name__.lower()
    
    if is_nonlinParamSys and 'paramInt' in params:
        # MATLAB: H = sys.hessian(sys.linError.p.x, sys.linError.p.u,params.paramInt);
        H = sys.hessian(sys.linError.p.x, sys.linError.p.u, params['paramInt'])
    else:
        # MATLAB: H = sys.hessian(sys.linError.p.x, sys.linError.p.u);
        H = sys.hessian(sys.linError.p.x, sys.linError.p.u)
    
    # calculate the quadratic map == static second order error
    # MATLAB: errorSecOrdStat = 0.5*quadMap(Z, H);
    errorSecOrdStat = 0.5 * Z.quadMap(H)
    
    # third-order error
    # MATLAB: if options.tensorOrder >= 4
    if options['tensorOrder'] >= 4:
        
        # set handle to correct file
        # MATLAB: sys = setThirdOrderTensor(sys,'standard');
        sys = sys.setThirdOrderTensor('standard')
       
        # reduce the order of the reachable set to speed-up the computations 
        # for cubic multiplication
        # MATLAB: if isfield(options,'errorOrder3')
        if 'errorOrder3' in options:
            
            # MATLAB: Rred = reduce(Rred,options.reductionTechnique,options.errorOrder3);
            Rred = Rred.reduce(options['reductionTechnique'], options['errorOrder3'])
            
            # MATLAB: Rdelta = reduce(Rdelta,options.reductionTechnique,options.errorOrder3);
            Rdelta = Rdelta.reduce(options['reductionTechnique'], options['errorOrder3'])
           
            # MATLAB: Z = cartProd(Rred,params.U);
            Z = Rred.cartProd_(params['U'])
            
            # MATLAB: Zdelta3 = cartProd(Rdelta,params.U);
            Zdelta3 = Rdelta.cartProd_(params['U'])
           
        else:
            # MATLAB: Zdelta3 = Zdelta;
            Zdelta3 = Zdelta
        
        # calculate the third-order tensor
        # MATLAB: if isa(sys,'nonlinParamSys')
        if is_nonlinParamSys and 'paramInt' in params:
            # MATLAB: [T,ind3] = sys.thirdOrderTensor(sys.linError.p.x, sys.linError.p.u, params.paramInt);
            T, ind3 = sys.thirdOrderTensor(sys.linError.p.x, sys.linError.p.u, params['paramInt'])
        else:
            # MATLAB: [T,ind3] = sys.thirdOrderTensor(sys.linError.p.x, sys.linError.p.u);
            T, ind3 = sys.thirdOrderTensor(sys.linError.p.x, sys.linError.p.u)
        
        # calculate the cubic map == static third-order error
        # MATLAB: errorThirdOrdStat = 1/6 * cubMap(Z,T,ind3);
        errorThirdOrdStat = (1.0 / 6.0) * Z.cubMap(T, ind3)
        
        # calculate the overall static linearization error
        # MATLAB: if isa(Z,'polyZonotope') || isa(Z,'conPolyZono')
        from cora_python.contSet.polyZonotope import PolyZonotope
        from cora_python.contSet.conPolyZono import ConPolyZono
        
        if isinstance(Z, (PolyZonotope, ConPolyZono)):
            # MATLAB: errorStat = exactPlus(errorSecOrdStat,errorThirdOrdStat);
            errorStat = errorSecOrdStat.exactPlus(errorThirdOrdStat)
        else:
            # MATLAB: errorStat = errorSecOrdStat + errorThirdOrdStat;
            errorStat = errorSecOrdStat + errorThirdOrdStat
        
    else:
        
        # MATLAB: errorStat = errorSecOrdStat;
        errorStat = errorSecOrdStat
    
    # reduce the complexity of the set of static errors
    # MATLAB: errorStat = reduce(errorStat,options.reductionTechnique,options.intermediateOrder);
    errorStat = errorStat.reduce(options['reductionTechnique'], options['intermediateOrder'])
    
    return H, Zdelta, errorStat, T, ind3, Zdelta3

