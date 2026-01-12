"""
eventFcn - returns the event function results of a guard set of a 
   transition

TRANSLATED FROM: cora_matlab/hybridDynamics/@transition/eventFcn.m

Syntax:
    [value,isterminal,direction] = eventFcn(trans,x)

Inputs:
    trans - transition object
    x - system state

Outputs:
    value - value of the event function
    isterminal - specifies if the simulation stops if an event turns zero
    direction - specifies if the value of the event function has to 
                turn from negative to positive or the other way round

Example:
    -

Authors:       Matthias Althoff (MATLAB)
Written:       07-May-2007 (MATLAB)
Last update:   07-September-2007 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Tuple
import numpy as np

if TYPE_CHECKING:
    from .transition import Transition


def eventFcn(trans: 'Transition', x: np.ndarray) -> Tuple[np.ndarray, bool, int]:
    """
    Returns the event function results of a guard set of a transition.
    
    Args:
        trans: transition object
        x: system state
    
    Returns:
        Tuple containing:
            value: value of the event function
            isterminal: specifies if the simulation stops if an event turns zero
            direction: specifies if the value of the event function has to 
                      turn from negative to positive (-1) or the other way round (1)
    
    Note:
        MATLAB code shows:
        % [value,isterminal,direction] = eventFcn(trans.guard,x,0);
        % [value,isterminal,direction] = eventFcn(trans.guard,x,1);
        % [value,isterminal,direction] = eventFcn(trans.guard,x,-1);
        The final line uses -1, so we use that.
    """
    # MATLAB: [value,isterminal,direction] = eventFcn(trans.guard,x,-1);
    # Call eventFcn on the guard set with direction -1
    # The guard set should have an eventFcn method
    if hasattr(trans.guard, 'eventFcn'):
        value, isterminal, direction = trans.guard.eventFcn(x, -1)
    else:
        # Fallback: try to import and call eventFcn from contSet
        from cora_python.contSet.contSet.eventFcn import eventFcn as contSet_eventFcn
        value, isterminal, direction = contSet_eventFcn(trans.guard, x, -1)
    
    return value, isterminal, direction

