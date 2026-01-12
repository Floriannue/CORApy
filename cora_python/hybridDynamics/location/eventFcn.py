"""
eventFcn - returns the handle of the event function for a location

TRANSLATED FROM: cora_matlab/hybridDynamics/@location/eventFcn.m

Syntax:
    han = eventFcn(loc)

Inputs:
    loc - location object

Outputs:
    han - event function handle (callable function)

Example:
    -

Authors:       Matthias Althoff (MATLAB)
Written:       07-May-2007 (MATLAB)
Last update:   06-June-2011, 17-October-2013 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Callable, Tuple
import numpy as np

if TYPE_CHECKING:
    from .location import Location


def eventFcn(loc: 'Location') -> Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns the handle of the event function for a location.
    
    The event function combines:
    - Invariant events (if invariant is not empty)
    - Guard events from all transitions
    
    Args:
        loc: location object
    
    Returns:
        Callable function that takes (t, x) and returns (value, isterminal, direction)
    """
    from cora_python.contSet.emptySet.emptySet import EmptySet
    
    def f(t: float, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standard syntax for event functions as needed in simulations.
        
        Args:
            t: time
            x: system state
        
        Returns:
            Tuple of (value, isterminal, direction)
        """
        # Get result of invariant events
        # MATLAB: if ~representsa_(loc.invariant,'emptySet',eps)
        from cora_python.contSet.contSet.representsa_ import representsa_
        
        if not representsa_(loc.invariant, 'emptySet', np.finfo(float).eps):
            # MATLAB: [value,isterminal,direction] = eventFcn(loc.invariant,x,1);
            if hasattr(loc.invariant, 'eventFcn'):
                value, isterminal, direction = loc.invariant.eventFcn(x, 1)
            else:
                # Try to import based on type
                from cora_python.contSet.polytope.polytope import Polytope
                if isinstance(loc.invariant, Polytope):
                    from cora_python.contSet.polytope.eventFcn import eventFcn as polytope_eventFcn
                    value, isterminal, direction = polytope_eventFcn(loc.invariant, x, 1)
                else:
                    raise AttributeError(f"Invariant type {type(loc.invariant)} does not have eventFcn method")
        else:
            # MATLAB: value = []; isterminal = []; direction = [];
            value = np.array([])
            isterminal = np.array([])
            direction = np.array([])
        
        # Retrieve system dimension
        # MATLAB: n=length(value);
        n = len(value)
        
        # Get result of guard events
        # MATLAB: for i=1:length(loc.transition)
        for i in range(len(loc.transition)):
            # MATLAB: if ~isemptyobject(loc.transition(i))
            if not loc.transition[i].isemptyobject():
                # MATLAB: [resValue,resIsterminal,resDirection] = eventFcn(loc.transition(i),x);
                resValue, resIsterminal, resDirection = loc.transition[i].eventFcn(x)
                eventLength = len(resValue)
                
                # MATLAB: indices = n+1:n+eventLength;
                # Python uses 0-based indexing, so indices = n:n+eventLength
                indices = list(range(n, n + eventLength))
                
                # MATLAB: value(indices,1) = resValue;
                # Extend arrays to accommodate new values
                if len(value) == 0:
                    value = resValue
                    isterminal = resIsterminal
                    direction = resDirection
                else:
                    # Concatenate new values
                    value = np.concatenate([value, resValue])
                    isterminal = np.concatenate([isterminal, resIsterminal])
                    direction = np.concatenate([direction, resDirection])
                
                # MATLAB: n = n+eventLength;
                n = n + eventLength
        
        return value, isterminal, direction
    
    # Return function handle
    return f

