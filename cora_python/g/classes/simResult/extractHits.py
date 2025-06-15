"""
extractHits - extracts time and states where guard intersection happened
    default: all guard intersections, can be set to specific location
    before the transition; note that we have to return cell-arrays for the
    state vectors before and after jumping since subsequent locations need
    not have the same number of states

Syntax:
    [tHit,xHit,xHit_] = extractHits(simRes)
    [tHit,xHit,xHit_] = extractHits(simRes,locIDstart)
    [tHit,xHit,xHit_] = extractHits(simRes,locIDstart,locIDend)

Inputs:
    simRes - simResult object
    locIDstart - (optional) location vector where transition was triggered
    locIDend - (optional) location vector to where state transitioned

Outputs:
    tHit - double vector of switching times (1xs)
    xHit - cell-arrays of state vectors before switching (1xs)
    xHit_ - cell-arrays of state vectors after switching (1xs)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: simResult, hybridAutomaton/simulateRandom
"""

from typing import TYPE_CHECKING, List, Tuple, Optional, Union
import numpy as np

if TYPE_CHECKING:
    from .simResult import SimResult

def extractHits(simRes: 'SimResult', 
                locIDstart: Optional[Union[int, np.ndarray]] = None,
                locIDend: Optional[Union[int, np.ndarray]] = None) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Extracts time and states where guard intersection happened.
    
    Args:
        simRes: simResult object
        locIDstart: (optional) location vector where transition was triggered
        locIDend: (optional) location vector to where state transitioned
        
    Returns:
        Tuple containing:
        - tHit: double vector of switching times
        - xHit: list of state vectors before switching
        - xHit_: list of state vectors after switching
    """
    # empty object or all continuous-time simulation runs
    if simRes.isemptyobject() or all(
        len(simRes.loc) == 1 and simRes.loc[0] == 0 
        for simRes in (simRes if isinstance(simRes, list) else [simRes])
    ):
        return np.array([]), [], []
    
    tHit = []
    xHit = []
    xHit_ = []
    
    # handle single simResult or list of simResults
    simRes_list = simRes if isinstance(simRes, list) else [simRes]
    
    # loop over all trajectories
    for r in range(len(simRes_list)):
        simRes_r = simRes_list[r]
        
        # shortened code for getting all hits
        if locIDstart is None and locIDend is None:
            # Get switching times (first time point of each segment after the first)
            for j in range(1, len(simRes_r.t)):
                tHit.append(simRes_r.t[j][0])
                # State before jump (last state of previous segment)
                xHit.append(simRes_r.x[j-1][-1, :])
                # State after jump (first state of current segment)
                xHit_.append(simRes_r.x[j][0, :])
            continue
        
        # longer version if locations have to be checked
        for j in range(len(simRes_r.t) - 1):
            # check if location before and after jump matches user input
            loc_before = simRes_r.loc[j] if hasattr(simRes_r, 'loc') else 0
            loc_after = simRes_r.loc[j+1] if hasattr(simRes_r, 'loc') else 0
            
            startcond = (locIDstart is None or 
                        np.array_equal(np.atleast_1d(loc_before), np.atleast_1d(locIDstart)))
            endcond = (locIDend is None or 
                      np.array_equal(np.atleast_1d(loc_after), np.atleast_1d(locIDend)))
            
            if startcond and endcond:
                # time before and after jump are equal so doesn't matter
                tHit.append(simRes_r.t[j][-1])
                # state vector before jump
                xHit.append(simRes_r.x[j][-1, :])
                # state vector after jump
                xHit_.append(simRes_r.x[j+1][0, :])
    
    return np.array(tHit), xHit, xHit_ 