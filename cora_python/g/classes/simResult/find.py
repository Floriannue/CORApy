"""
find - get simRes object that satisfy a certain condition

Syntax:
    res = find(simRes, prop, val)

Inputs:
    simRes - simResult object
    prop - property for condition ('location', 'time')
    val - value for property

Outputs:
    res - all parts of simRes trajectories that satisfy the condition

Authors: Mark Wetzlinger, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 16-May-2023 (MATLAB)
Last update: 10-April-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Any, Union


def find(simRes, prop: str, val: Any):
    """
    Get simRes object that satisfy a certain condition
    
    Args:
        simRes: simResult object
        prop: Property for condition ('location', 'time')
        val: Value for property
        
    Returns:
        All parts of simRes trajectories that satisfy the condition
    """
    # Validate inputs
    valid_props = ['location', 'time']
    if prop not in valid_props:
        raise ValueError(f"Property must be one of: {valid_props}")
    
    if prop == 'location':
        # Get all simRes trajectories with a given location number
        
        # Quick check: all trajectories from continuous-time system?
        if isinstance(val, int) and val == 0:
            if isinstance(simRes.loc, int) and simRes.loc == val:
                return simRes
            elif isinstance(simRes.loc, list) and all(loc == val for loc in simRes.loc):
                return simRes
        
        # Create new simResult with matching trajectories
        from .simResult import SimResult
        
        new_x = []
        new_t = []
        new_y = []
        new_a = []
        new_loc = []
        
        # Handle case where loc is a single value for all trajectories
        if isinstance(simRes.loc, int):
            if simRes.loc == val:
                return simRes
            else:
                return SimResult()  # Empty result
        
        # Handle case where loc is a list
        elif isinstance(simRes.loc, list):
            for i, loc in enumerate(simRes.loc):
                if loc == val:
                    if i < len(simRes.x):
                        new_x.append(simRes.x[i])
                    if i < len(simRes.t):
                        new_t.append(simRes.t[i])
                    if i < len(simRes.y):
                        new_y.append(simRes.y[i])
                    if i < len(simRes.a):
                        new_a.append(simRes.a[i])
                    new_loc.append(loc)
        
        return SimResult(new_x, new_t, new_loc, new_y, new_a)
    
    elif prop == 'time':
        # Get all simRes trajectories within a given time interval
        from ....contSet.interval.interval import Interval
        
        # Convert val to interval if it's not already
        if not isinstance(val, Interval):
            if isinstance(val, (int, float)):
                # Single time point - create small interval around it
                val = Interval(val - 1e-10, val + 1e-10)
            elif isinstance(val, (list, tuple, np.ndarray)) and len(val) == 2:
                val = Interval(val[0], val[1])
            else:
                raise ValueError("Time value must be a number, interval, or [start, end] pair")
        
        from .simResult import SimResult
        
        new_x = []
        new_t = []
        new_y = []
        new_a = []
        new_loc = []
        
        # Process each trajectory
        for i in range(len(simRes.x)):
            if i < len(simRes.t):
                t_traj = simRes.t[i]
                x_traj = simRes.x[i]
                
                # Find time indices within the interval
                mask = (t_traj >= val.inf) & (t_traj <= val.sup)
                
                if np.any(mask):
                    # Extract the portion within the time interval
                    new_t.append(t_traj[mask])
                    new_x.append(x_traj[mask])
                    
                    # Handle outputs
                    if i < len(simRes.y) and len(simRes.y[i]) > 0:
                        new_y.append(simRes.y[i][mask])
                    else:
                        new_y.append(np.array([]))
                    
                    # Handle algebraic variables
                    if i < len(simRes.a) and len(simRes.a[i]) > 0:
                        new_a.append(simRes.a[i][mask])
                    else:
                        new_a.append(np.array([]))
                    
                    # Handle location
                    if isinstance(simRes.loc, list) and i < len(simRes.loc):
                        new_loc.append(simRes.loc[i])
                    else:
                        new_loc.append(simRes.loc)
        
        # Set location appropriately
        if len(set(new_loc)) == 1:
            final_loc = new_loc[0] if new_loc else 0
        else:
            final_loc = new_loc
        
        return SimResult(new_x, new_t, final_loc, new_y, new_a)
    
    else:
        raise ValueError(f"Unknown property: {prop}") 