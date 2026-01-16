"""
priv_precompStatError_adaptive - precompute the second order static error
   along with Hessian matrix, with adaptive zonotope order reduction

Syntax:
    [H,Zdelta,errorStat,T,ind3,Zdelta3] = priv_precompStatError_adaptive(nlnsys,Rdelta,U,options)

Inputs:
    nlnsys - nonlinear system object
    Rdelta - shifted reachable set at the beginning of the time step
    U - input set
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

References: 
  [1] M. Althoff et al. "Reachability Analysis of Nonlinear Systems with 
      Uncertain Parameters using Conservative Linearization"
  [2] M. Althoff et al. "Reachability analysis of nonlinear systems using 
      conservative polynomialization and non-convex sets"

See also: linReach, linError_*, preCompStatError

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       --- (MATLAB)
Last update:   --- (MATLAB)
Python translation: 2025
"""

from typing import Any, Dict, Tuple, Optional, List
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.conPolyZono import ConPolyZono
from cora_python.contSet.zonotope.quadMap import quadMap


def priv_precompStatError_adaptive(nlnsys: Any, Rdelta: Any, U: Any, options: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Precompute second-order static error and Hessian.
    """
    # precompute the second order static error along with hessian matrix
    # MATLAB: nlnsys = setHessian(nlnsys,'standard');
    nlnsys = nlnsys.setHessian('standard')

    # initialize output arguments
    T = []
    ind3 = []
    Zdelta3 = []

    # reduce the reachable set for the initial time point
    # MATLAB: Rred = reduce(Rdelta,'adaptive',sqrt(options.redFactor));
    Rred_res = Rdelta.reduce('adaptive', np.sqrt(options['redFactor']))
    Rred = Rred_res[0] if isinstance(Rred_res, tuple) else Rred_res

    # over-approximation of the reachable set at the initial time point
    # MATLAB: Rdelta = reduce(zonotope(Rdelta),'adaptive',sqrt(options.redFactor));
    Rdelta_zono = Zonotope(Rdelta)
    Rdelta_res = Rdelta_zono.reduce('adaptive', np.sqrt(options['redFactor']))
    Rdelta_zono = Rdelta_res[0] if isinstance(Rdelta_res, tuple) else Rdelta_res

    # extend the sets by the input sets
    # MATLAB: Z = cartProd(Rred,U); Zdelta = cartProd(Rdelta,U);
    Z = Rred.cartProd_(U)
    Zdelta = Rdelta_zono.cartProd_(U)

    # calculate the hessian tensor
    # MATLAB: H = nlnsys.hessian(nlnsys.linError.p.x, nlnsys.linError.p.u);
    H = nlnsys.hessian(nlnsys.linError.p.x, nlnsys.linError.p.u)

    # calculate the quadratic map == static second-order error
    # MATLAB: errorSecOrdStat = 0.5*quadMap(Z,H);
    errorSecOrdStat = 0.5 * quadMap(Z, H)

    # reduce the order of the set of static errors
    # MATLAB: errorStat = reduce(errorSecOrdStat,'adaptive',sqrt(options.redFactor));
    errorStat_res = errorSecOrdStat.reduce('adaptive', np.sqrt(options['redFactor']))
    errorStat = errorStat_res[0] if isinstance(errorStat_res, tuple) else errorStat_res

    return H, Zdelta, errorStat, T, ind3, Zdelta3
