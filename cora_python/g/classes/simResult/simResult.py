"""
simResult - class that stores simulation results

This class stores the results of simulated trajectories including states,
outputs, algebraic variables, time points, and locations.

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 29-May-2020 (MATLAB)
Last update: 15-October-2024 (MATLAB)
Python translation: 2025
"""

from typing import List, Optional, Union, Any
import numpy as np


class SimResult:
    """
    Class that stores simulation results
    
    This class stores the results of simulated trajectories including:
    - States of simulated trajectories
    - Outputs of simulated trajectories  
    - Algebraic variables (for DAE systems)
    - Time points for trajectories
    - Location indices (for hybrid systems)
    
    Properties:
        x: Cell-array storing states of simulated trajectories,
           where each trajectory is a matrix of dimension [N,n]
        y: Cell-array storing outputs of simulated trajectories,
           where each trajectory is a matrix of dimension [N,o]
        a: Cell-array storing algebraic variables of simulated
           trajectories (only for nonlinDASys), where each trajectory
           is a matrix of dimension [N,p]
        t: Cell-array storing time points for simulated trajectories
           as vectors of dimension [N,1]
        loc: Index/indices of locations (0 for contDynamics)
    """
    
    def __init__(self, 
                 x: Optional[List[np.ndarray]] = None,
                 t: Optional[List[np.ndarray]] = None,
                 loc: Union[int, List[int]] = 0,
                 y: Optional[List[np.ndarray]] = None,
                 a: Optional[List[np.ndarray]] = None):
        """
        Constructor for simResult
        
        Args:
            x: List storing states of simulated trajectories
            t: List storing time points for simulated trajectories
            loc: Index/indices of locations (default: 0 for contDynamics)
            y: List storing outputs of simulated trajectories (optional)
            a: List storing algebraic variables (optional, for DAE systems)
        """
        # Initialize with empty lists if None provided
        self.x = x if x is not None else []
        self.t = t if t is not None else []
        self.y = y if y is not None else []
        self.a = a if a is not None else []
        self.loc = loc
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate that inputs are consistent"""
        # Check that x and t have same length if both provided
        if self.x and self.t:
            if len(self.x) != len(self.t):
                raise ValueError("x and t must have the same length")
        
        # Check that y has same length as x if provided
        if self.y and self.x:
            if len(self.y) != len(self.x):
                raise ValueError("y must have the same length as x")
        
        # Check that a has same length as x if provided
        if self.a and self.x:
            if len(self.a) != len(self.x):
                raise ValueError("a must have the same length as x")
        
        # Validate array dimensions
        for i, (x_traj, t_traj) in enumerate(zip(self.x, self.t)):
            if x_traj.shape[0] != t_traj.shape[0]:
                raise ValueError(f"Trajectory {i}: x and t must have same number of time steps")
    
    # Method implementations (imported from separate files)
    def find(self, prop: str, val: Any):
        """Find simResult objects that satisfy a certain condition"""
        from .find import find
        return find(self, prop, val)
    
    def add(self, other: 'SimResult') -> 'SimResult':
        """Add another simResult object"""
        from .add import add
        return add(self, other)
    
    def plot(self, *args, **kwargs):
        """Plot simulation results"""
        from .plot import plot
        return plot(self, *args, **kwargs)
    
    def plotOverTime(self, *args, **kwargs):
        """Plot simulation results over time"""
        from .plotOverTime import plotOverTime
        return plotOverTime(self, *args, **kwargs)
    
    def plotTimeStep(self, *args, **kwargs):
        """Plot simulation results at specific time steps"""
        from .plotTimeStep import plotTimeStep
        return plotTimeStep(self, *args, **kwargs)
    
    def isemptyobject(self) -> bool:
        """Check if simResult is empty"""
        from .isemptyobject import isemptyobject
        return isemptyobject(self)
    
    def extractHits(self, *args, **kwargs):
        """Extract hits from simulation results"""
        from .extractHits import extractHits
        return extractHits(self, *args, **kwargs)
    
    def monitorSTL(self, *args, **kwargs):
        """Monitor STL properties"""
        from .monitorSTL import monitorSTL
        return monitorSTL(self, *args, **kwargs)
    
    def printSimResult(self, *args, **kwargs):
        """Print simulation results"""
        from .printSimResult import printSimResult
        return printSimResult(self, *args, **kwargs)
    
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
    
    def __neg__(self):
        """Unary minus operation"""
        from .uminus import uminus
        return uminus(self)
    
    def __pos__(self):
        """Unary plus operation"""
        from .uplus import uplus
        return uplus(self)
    
    def __len__(self) -> int:
        """Return number of trajectories"""
        return len(self.x)
    
    def is_empty(self) -> bool:
        """Check if simResult is empty"""
        return self.isemptyobject()
    
    # String representation
    def __str__(self) -> str:
        """String representation"""
        n_traj = len(self.x)
        if n_traj == 0:
            return "SimResult: empty"
        
        n_states = self.x[0].shape[1] if self.x else 0
        n_outputs = self.y[0].shape[1] if self.y and len(self.y) > 0 else 0
        n_algebraic = self.a[0].shape[1] if self.a and len(self.a) > 0 else 0
        
        return (f"SimResult: {n_traj} trajectories, "
                f"{n_states} states, {n_outputs} outputs, "
                f"{n_algebraic} algebraic variables")
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__() 