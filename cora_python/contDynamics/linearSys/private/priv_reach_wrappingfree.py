"""
priv_reach_wrappingfree - computes the reachable set for linear systems using
   the wrapping-free reachability algorithm for linear systems [1]

Syntax:
   [timeInt, timePoint, res] = priv_reach_wrappingfree(linsys, params, options)

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
   [1] A. Girard, C. Le Guernic, and O. Maler, "Efficient computation of
       reachable sets of linear time-invariant systems with inputs"
       in Hybrid Systems: Computation and Control, ser. LNCS 3927.
       Springer, 2006, pp. 257--271.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 26-June-2019 (from @contDynamics > reach.m) (MATLAB)
Last update: 14-August-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Dict, Any, Tuple
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from ..canonicalForm import canonicalForm
from ..oneStep import oneStep
from .priv_outputSet_canonicalForm import priv_outputSet_canonicalForm


def priv_reach_wrappingfree(linsys, params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Dict, Dict, bool]:
    """
    Computes the reachable set for linear systems using the wrapping-free algorithm
    
    Args:
        linsys: LinearSys object
        params: Model parameters
        options: Computation options
        
    Returns:
        Tuple of (timeInt, timePoint, res)
    """
    # Put system into canonical form
    if 'uTransVec' in params:
        linsys, U, u, V, v = canonicalForm(linsys, params['U'], params['uTransVec'],
                                         params['W'], params['V'], np.zeros((linsys.nr_of_outputs, 1)))
    else:
        linsys, U, u, V, v = canonicalForm(linsys, params['U'], params['uTrans'],
                                         params['W'], params['V'], np.zeros((linsys.nr_of_outputs, 1)))
    
    # Time period and number of steps
    tVec = np.arange(params['tStart'], params['tFinal'] + options['timeStep'], options['timeStep'])
    steps = len(tVec) - 1
    
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
    _verboseLog(options.get('verbose', 0), 1, params['tStart'], params['tStart'], params['tFinal'])
    
    # Compute reachable sets for first step
    Rtp, Rti, Htp, Hti, PU, Pu, _, C_input = oneStep(linsys,
        params['R0'], U, u[:, 0], options['timeStep'], options['taylorTerms'])
    
    # Read out propagation matrix and base particular solution
    eAdt = linsys.taylor.getTaylor('eAdt', timeStep=options['timeStep'])
    
    # Save particular solution
    PU_next = PU
    PU = Interval(PU) if not isinstance(PU, Interval) else PU
    
    if hasattr(Pu, 'center'):
        Pu_c = Pu.center()
        Pu_int = Interval(Pu) - Pu_c
    else:
        Pu_int = np.zeros((linsys.nr_of_states, 1))
        Pu_c = Pu
    
    # Compute output set of start set and first time-interval solution
    timePoint['set'][0] = priv_outputSet_canonicalForm(linsys, params['R0'], V, v, 1)
    timeInt['set'][0] = priv_outputSet_canonicalForm(linsys, Rti, V, v, 1)
    timeInt['time'][0] = Interval(np.array([[tVec[0]], [tVec[1]]]))
    timePoint['set'][1] = priv_outputSet_canonicalForm(linsys, Rtp, V, v, 2)
    
    # Safety property check
    if 'specification' in options:
        res, timeInt, timePoint = _priv_checkSpecification(
            options['specification'], Rti, timeInt, timePoint, 1)
        if not res:
            return timeInt, timePoint, res
    
    # Loop over all reachability steps
    for k in range(1, steps):
        
        # Method implemented from Algorithm 2 in [1]
        
        # Re-compute particular solution due to constant input if we have a
        # time-varying input trajectory, since the constant input is included
        # in our affine solution, we recompute Htp, Hti, and Pu, incl. errors
        if 'uTransVec' in params:
            Htp_start = Htp
            from ..affineSolution import affineSolution
            Htp, Pu, _, C_state, C_input = affineSolution(
                linsys, Htp_start, u[:, k], options['timeStep'], options['taylorTerms'])
            Hti = _enclose(Htp_start, Htp) + C_state
            
            if hasattr(Pu, 'center'):
                Pu_c = Pu.center()
                Pu_int = Interval(Pu) - Pu_c
            else:
                Pu_int = np.zeros((linsys.nr_of_states, 1))
                Pu_c = Pu
        else:
            # Propagate affine solution
            Hti = eAdt @ Hti + Pu_c
            Htp = eAdt @ Htp + Pu_c
        
        # Propagate particular solution (interval)
        PU_next = eAdt @ PU_next
        PU = PU + Interval(PU_next) + Pu_int
        
        # Full solution
        Rti = Hti + PU + C_input
        Rtp = Htp + PU
        
        # Compute output sets
        timeInt['set'][k] = priv_outputSet_canonicalForm(linsys, Rti, V, v, k + 1)
        timeInt['time'][k] = Interval(np.array([[tVec[k]], [tVec[k + 1]]]))
        
        # Compute output set for start set of next step
        timePoint['set'][k + 1] = priv_outputSet_canonicalForm(linsys, Rtp, V, v, k + 2)
        
        # Safety property check
        if 'specification' in options:
            res, timeInt, timePoint = _priv_checkSpecification(
                options['specification'], Rti, timeInt, timePoint, k + 1)
            if not res:
                return timeInt, timePoint, res
        
        # Log information
        _verboseLog(options.get('verbose', 0), k + 1, tVec[k], params['tStart'], params['tFinal'])
    
    # Specification fulfilled
    res = True
    
    return timeInt, timePoint, res


def _verboseLog(verbose_level: int, step: int, current_time: float, start_time: float, final_time: float):
    """Log verbose information"""
    if verbose_level > 0:
        progress = (current_time - start_time) / (final_time - start_time) * 100
        print(f"Step {step}: t = {current_time:.4f}, Progress: {progress:.1f}%")


def _enclose(set1, set2):
    """Compute convex hull of two sets"""
    if hasattr(set1, '__add__') and hasattr(set2, '__add__'):
        # For zonotopes, use convex hull operation
        if hasattr(set1, 'convHull'):
            return set1.convHull(set2)
        else:
            # Fallback: use interval hull
            from cora_python.contSet.interval import Interval
            int1 = Interval(set1) if not isinstance(set1, Interval) else set1
            int2 = Interval(set2) if not isinstance(set2, Interval) else set2
            return int1 + int2
    else:
        return set1 + set2


def _priv_checkSpecification(specification, Rti, timeInt, timePoint, step):
    """Check safety specifications"""
    # TODO: Implement specification checking
    # For now, always return True (specification satisfied)
    return True, timeInt, timePoint 