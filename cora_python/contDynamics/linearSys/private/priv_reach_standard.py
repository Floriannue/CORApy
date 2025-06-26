"""
priv_reach_standard - computes the reachable set for linear systems using the
   standard (non-wrapping-free) reachability algorithm for linear systems

Syntax:
   [timeInt, timePoint, res] = priv_reach_standard(linsys, params, options)

Inputs:
   linsys - linearSys object
   params - model parameters
   options - options for the computation of reachable sets

Outputs:
   timeInt - array of time-interval reachable / output sets
   timePoint - array of time-point reachable / output sets
   res - true/false whether specification satisfied

Example:
   -

References:
   [1] A. Girard, "Reachability of uncertain linear systems using 
      zonotopes" in Hybrid Systems: Computation and Control, 
      ser. LNCS 3414. Springer, 2005, pp. 291--305.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 26-June-2019 (from @contDynamics > reach.m) (MATLAB)
Last update: 19-November-2022 (MW, modularize specification check) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Dict, Any, Tuple
from cora_python.contSet.interval import Interval
from .priv_outputSet_canonicalForm import priv_outputSet_canonicalForm
from .priv_checkSpecification import priv_checkSpecification
from cora_python.g.functions.verbose.verboseLog import verboseLog


def priv_reach_standard(linsys, params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Dict, Dict, bool]:
    """
    Computes the reachable set for linear systems using the standard algorithm
    
    Args:
        linsys: LinearSys object
        params: Model parameters
        options: Computation options
        
    Returns:
        Tuple of (timeInt, timePoint, res)
    """
    # Time period and number of steps
    # Use proper time vector generation to avoid floating-point accumulation errors
    steps = int(np.round((params['tFinal'] - params['tStart']) / options['timeStep']))
    tVec = np.linspace(params['tStart'], params['tFinal'], steps + 1)
    
    # Put system into canonical form: this allows us to compute the output sets
    # much more efficiently (see below)
    if 'uTransVec' in params:
        linsys, U, u, V, v = linsys.canonicalForm(params['U'], params['uTransVec'],
                                         params['W'], params['V'], np.zeros((linsys.nr_of_noises, 1)))
    else:
        linsys, U, u, V, v = linsys.canonicalForm(params['U'], params['uTrans'],
                                         params['W'], params['V'], np.zeros((linsys.nr_of_noises, 1)))
    
    # Check if time-varying inputs given
    isU = U.representsa('origin')
    
    # Initialize output variables for reachable sets and output sets
    timeInt = {
        'set': [None] * steps,
        'time': [None] * steps
    }
    timePoint = {
        'set': [None] * (steps + 1),
        'time': tVec.tolist()
    }
    
    # Log information
    verboseLog(options.get('verbose', 0), 1, params['tStart'], params['tStart'], params['tFinal'])
    
    # Compute reachable sets for first step
    Rtp, Rti, Htp, Hti, PU, Pu, _, C_input = linsys.oneStep(
        params['R0'], U, u[:, 0], options['timeStep'], options['taylorTerms'])
    
    # Read out propagation matrix and base particular solution
    eAdt = linsys.getTaylor('eAdt', {'timeStep': options['timeStep']})
    if not isU:
        PU_next = PU
    
    # Compute output set of start set and first time-interval solution
    timePoint['set'][0] = priv_outputSet_canonicalForm(linsys, params['R0'], V, v, 1)
    timeInt['set'][0] = priv_outputSet_canonicalForm(linsys, Rti, V, v, 1)
    timeInt['time'][0] = Interval(tVec[0], tVec[1])
    timePoint['set'][1] = priv_outputSet_canonicalForm(linsys, Rtp, V, v, 2)
    
    # Safety property check
    if 'specification' in options:
        res, timeInt, timePoint = priv_checkSpecification(
            options['specification'], Rti, timeInt, timePoint, 1)
        if not res:
            return timeInt, timePoint, res
    
    # Loop over all reachability steps
    for k in range(1, steps):
        
        # Method implemented from Algorithm 1 in [1]
        
        # Re-compute particular solution due to constant input if we have a
        # time-varying input trajectory, since the constant input is included
        # in our affine solution, we recompute Htp, Hti, and Pu, incl. errors
        if 'uTransVec' in params:
            Htp_start = Htp
            Htp, Pu, _, C_state, C_input = linsys.affineSolution(
                Htp_start, u[:, k], options['timeStep'], options['taylorTerms'])
            Hti = Htp_start.enclose(Htp) + C_state
        else:
            # Homogeneous solution, incl. reduction
            Htp = eAdt @ Htp + Pu
            Hti = eAdt @ Hti + Pu
        
        # Reduction
        Htp = Htp.reduce(options['reductionTechnique'], options['zonotopeOrder'])
        Hti = Hti.reduce(options['reductionTechnique'], options['zonotopeOrder'])
        
        if not isU:
            # Propagate particular solution (time-varying, centered at zero)
            PU_next = eAdt @ PU_next
            PU = (PU + PU_next).reduce(options['reductionTechnique'], options['zonotopeOrder'])
        
        # Compute reachable set
        Rti = Hti + PU + C_input
        Rtp = Htp + PU
        
        # Compute output set
        timeInt['set'][k] = priv_outputSet_canonicalForm(linsys, Rti, V, v, k + 1)
        timeInt['time'][k] = Interval(tVec[k], tVec[k + 1])
        
        # Compute output set for start set of next step
        timePoint['set'][k + 1] = priv_outputSet_canonicalForm(linsys, Rtp, V, v, k + 2)
        
        # Safety property check
        if 'specification' in options:
            res, timeInt, timePoint = priv_checkSpecification(
                options['specification'], Rti, timeInt, timePoint, k + 1)
            if not res:
                return timeInt, timePoint, res
        
        # Log information
        verboseLog(options.get('verbose', 0), k + 1, tVec[k], params['tStart'], params['tFinal'])
    
    # Specification fulfilled
    res = True
    
    return timeInt, timePoint, res


 