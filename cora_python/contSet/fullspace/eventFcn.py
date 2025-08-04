"""
eventFcn - Returns the results of an event function that detects if a 
   trajectory enters or leaves a fullspace (can never leave);
   this event function is needed, e.g. for matlab ODE-solvers

Syntax:
   [val,isterminal,direction] = eventFcn(fs,x,direction)

Inputs:
   fs - fullspace object
   x - system state
   direction - event if the state enters or leaves the set

Outputs:
   val - value of the event function
   isterminal - specifies if the simulation stops if an event turns zero
   direction - specifies if the value of the event function has to 
               turn from negative to positive or the other way round

Example:
   ---

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: polytope/eventFcn

Authors:       Mark Wetzlinger
Written:       14-December-2023 
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

import numpy as np

def eventFcn(fs, x, direction):
    """
    Returns the results of an event function that detects if a 
    trajectory enters or leaves a fullspace (can never leave);
    this event function is needed, e.g. for matlab ODE-solvers
    
    Args:
        fs: fullspace object
        x: system state
        direction: event if the state enters or leaves the set
        
    Returns:
        val: value of the event function
        isterminal: specifies if the simulation stops if an event turns zero
        direction: specifies if the value of the event function has to 
                   turn from negative to positive or the other way round
    """
    # compute value
    val = -np.inf * np.ones(fs.dimension)
    # always stop the integration when event detected
    isterminal = np.ones(len(val))
    # vectorize direction
    direction = np.ones(len(val)) * direction
    
    return val, isterminal, direction

# ------------------------------ END OF CODE ------------------------------ 