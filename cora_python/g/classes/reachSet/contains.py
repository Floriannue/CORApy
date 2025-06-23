"""
contains - checks if reachable set contains all simulated
    points of a set of system trajectory; not for hybrid systems

Syntax:
    res = contains(R,simRes)
    res = contains(R,simRes,type)
    res = contains(R,simRes,type,tol)

Inputs:
    R - object of class reachSet
    simRes - object of class simRes
    type - (optional) 'exact' or 'approx'
    tol - (optional) tolerance

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none
"""

from typing import TYPE_CHECKING, Optional, Union, List, Tuple
import numpy as np
from cora_python.contSet.interval.interval import Interval

if TYPE_CHECKING:
    from .reachSet import ReachSet
    from ..simResult.simResult import SimResult

def contains(R: 'ReachSet', simRes: 'SimResult', 
             type_: str = 'exact', tol: float = 1e-10) -> bool:
    """
    Checks if reachable set contains all simulated points of a set of system trajectory.
    
    Args:
        R: object of class reachSet
        simRes: object of class simResult
        type_: 'exact' or 'approx' (default: 'exact')
        tol: tolerance (default: 1e-10)
        
    Returns:
        bool: True if all simulation points are contained, False otherwise
        
    Raises:
        ValueError: if some points occur outside of the time covered by the reachable set
        NotImplementedError: if simRes is not a simResult object
    """
    # Check if simRes is a proper simResult object
    if not hasattr(simRes, 't') or not hasattr(simRes, 'x'):
        raise NotImplementedError("contains method currently only supports simResult objects")
    
    # Validate inputs
    if type_ not in ['exact', 'approx']:
        raise ValueError("type must be 'exact' or 'approx'")
    if tol < 0:
        raise ValueError("tolerance must be non-negative")
    
    # Handle list of simResults
    simRes_list = simRes if isinstance(simRes, list) else [simRes]
    
    # Total number of points: put all points of all trajectories into one big
    # list so that contains does not convert the same set into polytopes
    # multiple times (costly!)
    nrPoints = sum(sum(len(t_traj) for t_traj in sim.t) for sim in simRes_list)
    
    # logical indexing
    ptsContained = np.zeros(nrPoints, dtype=bool)
    ptsChecked = np.zeros(nrPoints, dtype=bool)
    
    # loop over locations of reachable set:
    # - purely-continuous: location always 0
    # - hybrid: different location numbers 1 ... max location
    allLoc = R.query('allLoc')
    
    # Handle case where allLoc is a single location
    if not isinstance(allLoc, list):
        allLoc = [allLoc]
    
    for i, iLoc in enumerate(allLoc):
        # read out reachable set of current location
        R_ = R.find('location', iLoc)
        
        # Handle case where R_ is not a list
        R_list = R_ if isinstance(R_, list) else [R_]
        
        # loop over all found branches
        for j in range(len(R_list)):
            R_j = R_list[j]
            
            # check whether time-point or time-interval reachable set given
            if (hasattr(R_j, 'timeInterval') and R_j.timeInterval and 
                'set' in R_j.timeInterval and R_j.timeInterval['set']):
                sets = R_j.timeInterval['set']
                time = R_j.timeInterval['time']
            else:
                sets = R_j.timePoint['set']
                time = R_j.timePoint['time']
            
            # loop over reachable sets
            for k in range(len(sets)):
                # for each reachable set, find corresponding simulation points
                # in the entire simResult object
                pts, ptsChecked_ = _find_points_in_interval(simRes_list, nrPoints, time[k], iLoc, tol)
                
                # check containment
                if len(pts) > 0:
                    # Use the set's contains method if available
                    if hasattr(sets[k], 'contains'):
                        contained = sets[k].contains(pts.T, type_, tol)
                    else:
                        # Fallback: assume all points are contained (conservative)
                        contained = np.ones(pts.shape[1], dtype=bool)
                    
                    ptsContained[ptsChecked_] = ptsContained[ptsChecked_] | contained
                
                # extend list of checked points
                ptsChecked = ptsChecked | ptsChecked_
                
                # exit if a point is found to be outside of the reachable set
                # (only if single branch given, otherwise sets from different
                # branches can cover the same time)
                if len(R_list) == 1 and not np.all(ptsContained[ptsChecked_]):
                    return False
    
    # check whether all points were checked
    if not np.all(ptsChecked):
        raise ValueError('Some points of the simulation occur outside of the time covered by the reachable set.')
    
    # check whether all points contained
    return np.all(ptsContained)


def _find_points_in_interval(simRes_list: List['SimResult'], nrPoints: int, 
                           time: Union[float, 'interval'], loc: int, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find points in simulation results that match the given time interval and location.
    
    Args:
        simRes_list: List of simResult objects
        nrPoints: Total number of points
        time: Time interval or time point
        loc: Location index
        tol: Tolerance for time matching
        
    Returns:
        Tuple of (points, ptsChecked) where:
        - points: Array of state vectors that match criteria
        - ptsChecked: Boolean array indicating which points were checked
    """
    # check whether time interval or time point given
    ti = isinstance(time, Interval)
    
    # init points and logical value for contained points
    pts = np.zeros((0, simRes_list[0].x[0].shape[1] if simRes_list[0].x else 2))
    ptsChecked = np.zeros(nrPoints, dtype=bool)
    
    # index for full list
    startIdx = 0
    
    # loop over all individual trajectories
    for r in range(len(simRes_list)):
        simRes_r = simRes_list[r]
        
        # loop over all individual parts
        for part in range(len(simRes_r.t)):
            # number of points in this part
            nrPointsPart = len(simRes_r.t[part])
            
            # skip non-matching locations
            sim_loc = simRes_r.loc if hasattr(simRes_r, 'loc') else 0
            if isinstance(sim_loc, list):
                sim_loc = sim_loc[part] if part < len(sim_loc) else 0
            
            if sim_loc == loc:
                # check which points match the correct time
                if ti:
                    # time points have to be contained in time interval
                    tempIdx = time.contains(simRes_r.t[part], 'exact', tol)
                else:
                    # time points have to be within tolerance of given time point
                    tempIdx = np.abs(simRes_r.t[part] - time) <= tol
                
                # append checked points
                ptsChecked[startIdx:startIdx+nrPointsPart] = tempIdx
                
                # append corresponding state vectors
                if np.any(tempIdx):
                    matching_states = simRes_r.x[part][tempIdx, :]
                    pts = np.vstack([pts, matching_states]) if pts.size > 0 else matching_states
            
            # shift start index
            startIdx += nrPointsPart
    
    return pts.T, ptsChecked 