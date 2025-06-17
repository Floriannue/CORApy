"""
initialSet - class that stores the initial set of a reachSet object

This class wraps the initial set and provides plotting functionality.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 01-March-2023 (MATLAB)
Python translation: 2025
"""

from typing import List, Any, Optional


class InitialSet:
    """
    Class that stores the initial set of a reachSet object
    
    This class wraps an initial set and provides plotting methods
    compatible with the MATLAB CORA library.
    
    Properties:
        set: The initial set object
    """
    
    def __init__(self, set_obj):
        """
        Constructor for InitialSet
        
        Args:
            set_obj: The initial set (should be a contSet object)
        """
        # Validate that set_obj is a contSet (if available)
        try:
            from ..contSet.contSet.contSet import ContSet
            if not isinstance(set_obj, ContSet):
                raise ValueError("set must be a contSet object")
        except ImportError:
            # If ContSet is not available, skip validation
            pass
        
        self.set = set_obj
    
    def plot(self, dims: Optional[List[int]] = None, **kwargs):
        """
        Plot the initial set
        
        Args:
            dims: Dimensions to plot (default: [0, 1] for Python 0-indexing)
            **kwargs: Additional plotting options
        
        Returns:
            Handle to the plot
        """
        if dims is None:
            dims = [0, 1]  # Python uses 0-based indexing
        
        # Extract plotting options for initial set
        try:
            from ...functions.verbose.plot.read_plot_options import readPlotOptions
            plot_options = readPlotOptions(kwargs, 'initialSet')
        except ImportError:
            try:
                from cora_python.g.functions.verbose.plot.read_plot_options import readPlotOptions
                plot_options = readPlotOptions(kwargs, 'initialSet')
            except ImportError:
                # Fallback: use kwargs directly
                plot_options = kwargs
        
        # Plot the underlying set
        return self.set.plot(dims, **plot_options)
    
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
        try:
            from ...functions.verbose.plot.read_plot_options import readPlotOptions
            plot_options = readPlotOptions(kwargs, 'initialSet')
        except ImportError:
            try:
                from cora_python.g.functions.verbose.plot.read_plot_options import readPlotOptions
                plot_options = readPlotOptions(kwargs, 'initialSet')
            except ImportError:
                # Fallback: use kwargs directly
                plot_options = kwargs
        
        # Project to specified dimension
        try:
            from ...contSet.contSet.project import project
            projected_set = project(self.set, [dim])
        except ImportError:
            try:
                from cora_python.contSet.contSet.project import project
                projected_set = project(self.set, [dim])
            except ImportError:
                # Fallback: use the set's project method if available
                if hasattr(self.set, 'project'):
                    projected_set = self.set.project([dim])
                else:
                    projected_set = self.set
        
        # Create time-extended set: [0; 1] * set
        import numpy as np
        time_matrix = np.array([[0], [1]])  # Time from 0 to 1
        
        # For zonotope: multiply time vector with set
        if hasattr(projected_set, 'c') and hasattr(projected_set, 'G'):
            # This is a zonotope-like object
            try:
                from ...contSet.zonotope.zonotope import Zonotope
            except ImportError:
                from cora_python.contSet.zonotope.zonotope import Zonotope
            
            # Create extended set by Cartesian product with time interval
            extended_center = np.kron(time_matrix, projected_set.c)
            extended_generators = np.kron(time_matrix, projected_set.G)
            
            extended_set = Zonotope(extended_center, extended_generators)
            
            # Plot in time-space coordinates [time, state]
            return extended_set.plot([0, 1], **plot_options)
        else:
            # For other set types, use generic approach
            return projected_set.plot([0], **plot_options)
    
    def __str__(self) -> str:
        """String representation"""
        return f"InitialSet containing: {self.set}"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"InitialSet({repr(self.set)})" 