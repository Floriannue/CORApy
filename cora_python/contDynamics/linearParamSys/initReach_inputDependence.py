"""
initReach_inputDependence - computes the continuous reachable continuous 
   for the first time step when the constant input is parameterized and
   correlated to the parameters of the system

Syntax:
    [sys, Rfirst, options] = initReach_inputDependence(sys, Rinit, params, options)

Inputs:
    sys - LinearParamSys object
    Rinit - initial reachable set
    params - model parameters (must contain 'Uconst', 'uTrans')
    options - options for the computation of the reachable set (must contain 'timeStep', 'taylorTerms', 
              'intermediateTerms', 'reductionTechnique', 'zonotopeOrder')

Outputs:
    sys - LinearParamSys object
    Rfirst - first reachable set (dict with 'tp' and 'ti' keys)
    options - options for the computation of the reachable set

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import Any, Dict, Tuple
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope import enclose
from cora_python.contSet.zonotope.reduce import reduce
# center is attached to Zonotope class, use object.center()
from .private.priv_mappingMatrix import priv_mappingMatrix
from .private.priv_highOrderMappingMatrix import priv_highOrderMappingMatrix
from .private.priv_tie import priv_tie
from .private.priv_inputSolution import priv_inputSolution
from .private.priv_dependentHomSol import priv_dependentHomSol


def initReach_inputDependence(sys: Any, Rinit: Any, params: Dict[str, Any], 
                              options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Computes the first reachable set for linear parametric systems with input dependence
    
    Args:
        sys: LinearParamSys object
        Rinit: Initial reachable set (Zonotope)
        params: Model parameters dict (must contain 'Uconst', 'uTrans')
        options: Options dict (must contain 'timeStep', 'taylorTerms', 'intermediateTerms',
                 'reductionTechnique', 'zonotopeOrder')
        
    Returns:
        sys: Modified LinearParamSys object
        Rfirst: Dict with 'tp' (time point) and 'ti' (time interval) reachable sets
        options: Options dict (unchanged)
    """
    # Store taylor terms and time step as object properties
    # MATLAB: sys.stepSize = options.timeStep;
    # MATLAB: sys.taylorTerms = options.taylorTerms;
    sys.stepSize = options['timeStep']
    sys.taylorTerms = options['taylorTerms']
    
    # Compute mapping matrix
    # MATLAB: sys = priv_mappingMatrix(sys,params,options);
    sys = priv_mappingMatrix(sys, params, options)
    
    # Compute high order mapping matrix
    # MATLAB: sys = priv_highOrderMappingMatrix(sys,options.intermediateTerms);
    sys = priv_highOrderMappingMatrix(sys, options['intermediateTerms'])
    
    # Compute time interval error (tie)
    # MATLAB: sys = priv_tie(sys);
    sys = priv_tie(sys)
    
    # Compute reachable set due to input
    # MATLAB: sys = priv_inputSolution(sys,params,options);
    sys = priv_inputSolution(sys, params, options)
    
    # Compute reachable set of first time interval
    # First time step homogeneous solution
    # MATLAB: Rhom_tp = priv_dependentHomSol(sys, Rinit, params.Uconst);
    Rhom_tp = priv_dependentHomSol(sys, Rinit, params['Uconst'])
    
    # Time interval solution
    # MATLAB: inputCorr = sys.inputF*sys.B*zonotope(params.uTrans + center(params.Uconst));
    uTrans_center = params['uTrans'] + params['Uconst'].center()
    inputCorr = sys.inputF * sys.B * Zonotope(uTrans_center, np.zeros((sys.nr_of_dims, 0)))
    
    # MATLAB: Rhom = enclose(Rinit,Rhom_tp) + sys.F*Rinit + inputCorr;
    Rhom = enclose(Rinit, Rhom_tp) + sys.F * Rinit + inputCorr
    
    # Total solution
    # MATLAB: Rtotal = Rhom + sys.RV;
    # MATLAB: Rtotal_tp = Rhom_tp + sys.RV;
    Rtotal = Rhom + sys.RV
    Rtotal_tp = Rhom_tp + sys.RV
    
    # Write results to reachable set struct Rfirst
    # MATLAB: Rfirst.tp = reduce(Rtotal_tp,options.reductionTechnique,options.zonotopeOrder);
    # MATLAB: Rfirst.ti = reduce(Rtotal,options.reductionTechnique,options.zonotopeOrder);
    Rfirst = {
        'tp': reduce(Rtotal_tp, options['reductionTechnique'], options['zonotopeOrder']),
        'ti': reduce(Rtotal, options['reductionTechnique'], options['zonotopeOrder'])
    }
    
    return sys, Rfirst, options
