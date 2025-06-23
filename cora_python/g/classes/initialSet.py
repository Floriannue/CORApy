"""
initialSet - class that stores the initial set of a reachSet object

This class wraps the initial set and provides plotting functionality.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 01-March-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import List, Any, Optional
from cora_python.contSet.contSet.contSet import ContSet
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.verbose.plot.read_plot_options import read_plot_options

class InitialSet:
    """
    Class that stores the initial set of a reachSet object
    
    This class wraps an initial set and provides plotting methods
    compatible with the MATLAB CORA library.
    
    Properties:
        set: The initial set object
    """
    
    def __init__(self, set_obj: ContSet):
        """
        Constructor for InitialSet
        
        Args:
            set_obj: The initial set (should be a contSet object)
        """
        # Validate that set_obj is a contSet (if available)
        if not isinstance(set_obj, ContSet):
            raise ValueError("set must be a contSet object")
        
        self.set = set_obj
    
    def plot(self, dims: Optional[List[int]] = None, **kwargs):
        """
        Plot the initial set
        
        Args:
            dims: Dimensions to plot (default: [0, 1] for Python 0-indexing)
            **kwargs: Additional plotting options including DisplayName for MATLAB compatibility
        
        Returns:
            Handle to the plot
        """
        if dims is None:
            dims = [0, 1]  # Python uses 0-based indexing
        
        # Just pass the original kwargs and let contSet.plot() handle all processing
        # This avoids double-processing issues
        return self.set.plot(dims, purpose='initialSet', **kwargs)
    
    def plotOverTime(self, dim: int = 0, **kwargs):
        """
        Plot the initial set over time
        
        Args:
            dim: Dimension to plot (default: 0 for Python 0-indexing)
            **kwargs: Additional plotting options
        
        Returns:
            Handle to the plot
        """
        # Validate input
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("dim must be a non-negative integer")
        
        # Extract plotting options for initial set
        plot_options = read_plot_options(kwargs, 'initialSet')

        
        # Project to specified dimension
        projected_set = self.set.project([dim])

        
        # For zonotope-like objects, create time-extended set
        if hasattr(projected_set, 'c') and hasattr(projected_set, 'G'):



            
            # Ensure we have proper arrays
            c = np.asarray(projected_set.c, dtype=float)
            G = np.asarray(projected_set.G, dtype=float)
            
            # Ensure center is a column vector
            if c.ndim == 1:
                c = c.reshape(-1, 1)
            
            # Create time interval zonotope: time ∈ [0, 1]
            # This represents the interval from time 0 to time 1
            time_center = np.array([[0.5]])  # Center at t=0.5
            time_generator = np.array([[0.5]])  # ±0.5 gives [0, 1]
            
            # Create the time-extended set by Cartesian product
            # Result will be 2D: [time, projected_dimension]
            extended_center = np.vstack([time_center, c])
            
            # Create generators: time generator + projected generators
            if G.size > 0:
                # Add time generator and extend the original generators
                time_gen = np.vstack([time_generator, np.zeros((c.shape[0], 1))])
                extended_gens = []
                for i in range(G.shape[1]):
                    gen_i = np.vstack([np.zeros((1, 1)), G[:, i:i+1]])
                    extended_gens.append(gen_i)
                extended_gens.append(time_gen)
                extended_generators = np.hstack(extended_gens)
            else:
                # Only time generator
                extended_generators = np.vstack([time_generator, np.zeros((c.shape[0], 1))])
            
            # Create extended zonotope
            extended_set = Zonotope(extended_center, extended_generators)
            
            # Plot in time-space coordinates [time, state]
            return extended_set.plot([0, 1], **plot_options)
        else:
            # For other set types, create a simple line plot at t=0
            import matplotlib.pyplot as plt
            
            # Get the center or representative point
            if hasattr(projected_set, 'center'):
                point = projected_set.center()
            elif hasattr(projected_set, 'c'):
                point = projected_set.c
            else:
                # Fallback to zero
                point = np.array([[0.0]])
            
            if point.ndim == 1:
                point = point.reshape(-1, 1)
            
            # Plot as a vertical line at t=0
            time_points = [0, 1]
            state_points = [point[0, 0], point[0, 0]]
            
            line = plt.plot(time_points, state_points, **plot_options)
            return line[0] if line else None
    
    def __str__(self) -> str:
        """String representation"""
        return f"InitialSet containing: {self.set}"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"InitialSet({repr(self.set)})" 