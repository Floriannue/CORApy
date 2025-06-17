"""
reachSet - class for storing reachable sets

This class stores reachable sets computed by reachability analysis algorithms.
It contains both time-point and time-interval reachable sets.

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 29-May-2020 (MATLAB)
Last update: 15-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union


class _DotDict:
    """A dictionary that supports dot notation access"""
    def __init__(self, d):
        if d is None:
            d = {}
        for key, value in d.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def keys(self):
        return vars(self).keys()
    
    def items(self):
        return vars(self).items()
    
    def values(self):
        return vars(self).values()


class ReachSet:
    """
    Class for storing reachable sets
    
    This class stores the results of reachability analysis including
    time-point and time-interval reachable sets.
    
    Properties:
        timePoint: Object containing time-point reachable sets
                  - set: List of sets at time points
                  - time: List of time values
                  - error: List of error values (optional)
        timeInterval: Object containing time-interval reachable sets
                     - set: List of sets over time intervals
                     - time: List of time intervals
                     - error: List of error values (optional)
        parent: Index of parent reachable set (default: 0)
        loc: Index of location for hybrid systems (default: 0)
    """
    
    def __init__(self, timePoint: Optional[Dict] = None, timeInterval: Optional[Dict] = None, 
                 parent: int = 0, loc: int = 0):
        """
        Constructor for reachSet
        
        Args:
            timePoint: Time-point reachable sets
            timeInterval: Time-interval reachable sets
            parent: Index of parent reachable set (default: 0)
            loc: Index of location for hybrid systems (default: 0)
        """
        # Initialize empty instantiation
        if timePoint is None and timeInterval is None:
            self.timePoint = _DotDict({})
            self.timeInterval = _DotDict({})
            self.parent = 0
            self.loc = 0
            return
        
        # Validate inputs
        if timePoint is not None and not isinstance(timePoint, dict):
            raise ValueError("timePoint must be a dictionary")
        if timeInterval is not None and not isinstance(timeInterval, dict):
            raise ValueError("timeInterval must be a dictionary")
        if not isinstance(parent, int) or parent < 0:
            raise ValueError("parent must be a non-negative integer")
        if not isinstance(loc, int) or loc < 0:
            raise ValueError("loc must be a non-negative integer")
        
        self.timePoint = _DotDict(timePoint if timePoint is not None else {})
        self.timeInterval = _DotDict(timeInterval if timeInterval is not None else {})
        self.parent = parent
        self.loc = loc
    
    # Method implementations (imported from separate files)
    def query(self, property_name: str) -> Any:
        """Query reachable sets for specific properties"""
        from .query import query
        return query(self, property_name)
    
    def contains(self, simResult, type_: str = 'exact', tol: float = 1e-10) -> bool:
        """Check if simulation results are contained in reachable sets"""
        from .contains import contains
        return contains(self, simResult, type_, tol)
    
    def project(self, dims: List[int]) -> 'ReachSet':
        """Project reachable sets to lower-dimensional subspace"""
        from .project import project
        return project(self, dims)
    
    def find(self, prop: str, val: Any) -> 'ReachSet':
        """Find reachSet objects that satisfy a certain condition"""
        from .find import find
        return find(self, prop, val)
    
    def add(self, other: 'ReachSet') -> 'ReachSet':
        """Add another reachSet object"""
        from .add import add
        return add(self, other)
    
    def append(self, other: 'ReachSet') -> 'ReachSet':
        """Append another reachSet object at the end"""
        from .append import append
        return append(self, other)
    
    def children(self, R_list: List['ReachSet'], parent: int) -> List[int]:
        """Return a list of indices of the children of this parent node"""
        from .children import children
        return children(R_list, parent)
    
    def isequal(self, other: 'ReachSet', tol: float = 1e-12) -> bool:
        """Check if two reachSet objects are equal"""
        from .isequal import isequal
        return isequal(self, other, tol)
    
    def order(self) -> 'ReachSet':
        """Order the elements chronologically"""
        from .order import order
        return order(self)
    
    def shiftTime(self, delta: Union[int, float]) -> 'ReachSet':
        """Shift all sets by a scalar time delta"""
        from .shiftTime import shiftTime
        return shiftTime(self, delta)
    
    def plot(self, *args, **kwargs):
        """Plot reachable sets"""
        from .plot import plot
        return plot(self, *args, **kwargs)
    
    def plotOverTime(self, *args, **kwargs):
        """Plot reachable sets over time"""
        from .plotOverTime import plotOverTime
        return plotOverTime(self, *args, **kwargs)
    
    def plotTimeStep(self, *args, **kwargs):
        """Plot time step size over time"""
        from .plotTimeStep import plotTimeStep
        return plotTimeStep(self, *args, **kwargs)
    
    def plotAsGraph(self):
        """Plot branches of reachable set as graph"""
        from .plotAsGraph import plotAsGraph
        return plotAsGraph(self)
    
    def modelChecking(self, eq, alg: str = 'sampledTime', *args, **kwargs) -> bool:
        """Check if reachable set satisfies an STL formula"""
        from .modelChecking import modelChecking
        return modelChecking(self, eq, alg, *args, **kwargs)
    
    def isemptyobject(self) -> bool:
        """Check if reachSet is empty"""
        from .isemptyobject import isemptyobject
        return isemptyobject(self)
    
    @property
    def R0(self):
        """Get initial set (equivalent to MATLAB's R.R0 property)"""
        try:
            from ...contSet.emptySet import EmptySet
            from ..initialSet import InitialSet
        except ImportError:
            try:
                from cora_python.contSet.emptySet import EmptySet
                from cora_python.g.classes.initialSet import InitialSet
            except ImportError:
                # Fallback - create simple wrapper
                class InitialSet:
                    def __init__(self, set_obj):
                        self.set = set_obj
                    def plot(self, *args, **kwargs):
                        return self.set.plot(*args, **kwargs)
                
                class EmptySet:
                    def __init__(self, n=3):
                        self.n = n
                    def plot(self, *args, **kwargs):
                        pass
        
        if (not hasattr(self.timePoint, 'set') or 
            not self.timePoint.set or 
            len(self.timePoint.set) == 0):
            # Return empty set for plotting compatibility
            return InitialSet(EmptySet(3))
        else:
            # Return the first time-point set wrapped in InitialSet
            return InitialSet(self.timePoint.set[0])
    
    def plus(self, other: 'ReachSet') -> 'ReachSet':
        """Addition operation"""
        from .plus import plus
        return plus(self, other)
    
    def minus(self, other: 'ReachSet') -> 'ReachSet':
        """Subtraction operation"""
        from .minus import minus
        return minus(self, other)
    
    def times(self, other) -> 'ReachSet':
        """Element-wise multiplication operation"""
        from .times import times
        return times(self, other)
    
    def mtimes(self, other) -> 'ReachSet':
        """Matrix multiplication operation"""
        from .mtimes import mtimes
        return mtimes(self, other)
    
    def uminus(self) -> 'ReachSet':
        """Unary minus operation"""
        from .uminus import uminus
        return uminus(self)
    
    def uplus(self) -> 'ReachSet':
        """Unary plus operation"""
        from .uplus import uplus
        return uplus(self)
    
    def eq(self, other: 'ReachSet') -> bool:
        """Equality comparison"""
        from .eq import eq
        return eq(self, other)
    
    def ne(self, other: 'ReachSet') -> bool:
        """Inequality comparison"""
        from .ne import ne
        return ne(self, other)
    
    # Operator overloads
    def __add__(self, other):
        """Addition operation"""
        from .plus import plus
        return plus(self, other)
    
    def __radd__(self, other):
        """Reverse addition operation"""
        from .plus import plus
        return plus(other, self)
    
    def __sub__(self, other):
        """Subtraction operation"""
        from .minus import minus
        return minus(self, other)
    
    def __rsub__(self, other):
        """Reverse subtraction operation"""
        from .minus import minus
        return minus(other, self)
    
    def __mul__(self, other):
        """Element-wise multiplication operation"""
        from .times import times
        return times(self, other)
    
    def __rmul__(self, other):
        """Reverse element-wise multiplication operation"""
        from .times import times
        return times(other, self)
    
    def __matmul__(self, other):
        """Matrix multiplication operation"""
        from .mtimes import mtimes
        return mtimes(self, other)
    
    def __rmatmul__(self, other):
        """Reverse matrix multiplication operation"""
        from .mtimes import mtimes
        return mtimes(other, self)
    
    def __eq__(self, other):
        """Equality comparison"""
        from .eq import eq
        return eq(self, other)
    
    def __ne__(self, other):
        """Inequality comparison"""
        from .ne import ne
        return ne(self, other)
    
    def __neg__(self):
        """Unary minus operation"""
        from .uminus import uminus
        return uminus(self)
    
    def __pos__(self):
        """Unary plus operation"""
        from .uplus import uplus
        return uplus(self)
    
    # Static methods
    @staticmethod
    def initReachSet(timePoint: Dict, timeInterval: Optional[Dict] = None) -> 'ReachSet':
        """Initialize a reachSet object"""
        from .initReachSet import initReachSet
        return initReachSet(timePoint, timeInterval)
    
    # String representation
    def __str__(self) -> str:
        """String representation of reachSet"""
        tp_sets = len(self.timePoint.get('set', []))
        ti_sets = len(self.timeInterval.get('set', []))
        return f"reachSet (time-point sets: {tp_sets}, time-interval sets: {ti_sets})"
    
    def __repr__(self) -> str:
        """Detailed string representation of reachSet"""
        return self.__str__()
    
    def __len__(self) -> int:
        """Return the number of branches in the reachSet"""
        # For now, we consider a single reachSet as having length 1
        # In MATLAB, this would be the number of branches/elements in an array of reachSets
        return 1 