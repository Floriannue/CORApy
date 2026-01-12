"""
guard2polytope - convert the guard set of a transition to a polytope

TRANSLATED FROM: cora_matlab/hybridDynamics/@transition/guard2polytope.m

Syntax:
    trans = guard2polytope(trans)

Inputs:
    trans - transition object

Outputs:
    trans - modified transition object

Example:
    -

Authors:       Niklas Kochdumper (MATLAB)
Written:       16-May-2018 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .transition import Transition


def guard2polytope(trans: 'Transition') -> 'Transition':
    """
    Convert the guard set of a transition to a polytope.
    
    Args:
        trans: transition object
    
    Returns:
        Transition: modified transition object with guard converted to polytope
    """
    from cora_python.contSet.polytope.polytope import Polytope
    from cora_python.contSet.levelSet.levelSet import LevelSet
    
    # MATLAB: if ~isa(trans.guard,'levelSet') && ~isa(trans.guard,'polytope')
    if not isinstance(trans.guard, LevelSet) and not isinstance(trans.guard, Polytope):
        # MATLAB: trans.guard = polytope(trans.guard);
        # Convert to polytope (handles Interval and other set types)
        trans.guard = Polytope(trans.guard)
    
    return trans

