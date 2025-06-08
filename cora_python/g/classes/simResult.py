"""
simResult - class that stores simulation results

This class stores the results of simulated trajectories including states,
outputs, algebraic variables, time points, and locations.

Authors: Florian Nüssel (Python implementation)
Date: 2025-01-08
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
    
    Attributes:
        x (List): Cell-array storing states of simulated trajectories,
                 where each trajectory is a matrix of dimension [N,n]
        y (List): Cell-array storing outputs of simulated trajectories,
                 where each trajectory is a matrix of dimension [N,o]
        a (List): Cell-array storing algebraic variables of simulated
                 trajectories (only for nonlinDASys), where each trajectory
                 is a matrix of dimension [N,p]
        t (List): Cell-array storing time points for simulated trajectories
                 as vectors of dimension [N,1]
        loc (Union[int, List]): Index/indices of locations (0 for contDynamics)
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
    
    def __len__(self) -> int:
        """Return number of trajectories"""
        return len(self.x)
    
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
    
    def is_empty(self) -> bool:
        """Check if simResult is empty"""
        return len(self.x) == 0
    
    def __add__(self, other):
        """Addition operator for simResult objects"""
        if not isinstance(other, SimResult):
            raise TypeError("Can only add SimResult objects")
        
        # Combine trajectories
        new_x = self.x + other.x
        new_t = self.t + other.t
        new_y = self.y + other.y
        new_a = self.a + other.a
        
        # Handle locations
        if isinstance(self.loc, list) and isinstance(other.loc, list):
            new_loc = self.loc + other.loc
        elif isinstance(self.loc, list):
            new_loc = self.loc + [other.loc]
        elif isinstance(other.loc, list):
            new_loc = [self.loc] + other.loc
        else:
            new_loc = [self.loc, other.loc]
        
        return SimResult(new_x, new_t, new_loc, new_y, new_a)
    
    def __mul__(self, scalar):
        """Scalar multiplication of states"""
        if not isinstance(scalar, (int, float, np.number)):
            raise TypeError("Can only multiply by scalar")
        
        new_x = [x * scalar for x in self.x]
        new_y = [y * scalar for y in self.y] if self.y else []
        new_a = [a * scalar for a in self.a] if self.a else []
        
        return SimResult(new_x, self.t.copy(), self.loc, new_y, new_a)
    
    def __rmul__(self, scalar):
        """Right scalar multiplication"""
        return self.__mul__(scalar)
    
    def __matmul__(self, matrix):
        """Matrix multiplication of states"""
        matrix = np.asarray(matrix)
        
        new_x = [x @ matrix.T for x in self.x]
        new_y = [y @ matrix.T for y in self.y] if self.y else []
        
        return SimResult(new_x, self.t.copy(), self.loc, new_y, self.a.copy() if self.a else []) 