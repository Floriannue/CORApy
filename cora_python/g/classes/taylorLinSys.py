"""
taylorLinSys - helper class for storing Taylor series computations for linear systems

This class stores precomputed matrices and values for Taylor series expansions
used in reachability analysis of linear systems.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 07-May-2007 (MATLAB)
Last update: 25-July-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Dict, Any, Optional, List
from scipy.linalg import expm, inv
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TaylorLinSys:
    """
    Helper class for storing Taylor series computations for linear systems
    
    This class precomputes and stores matrices needed for reachability analysis
    including exponential matrices, Taylor series powers, and error bounds.
    
    Properties:
        A: System matrix
        timeStep: Current time step size
        eAt: Matrix exponential e^(A*timeStep)
        eAdt: Alias for eAt
        Ainv: Inverse of A (if it exists)
        powers: List of matrix powers A^i
        error: Error bound matrix
        F: Correction matrix for state
        G: Correction matrix for input
        V: Input set
        RV: Particular solution due to time-varying inputs
        Rtrans: Particular solution due to constant inputs
        inputCorr: Input correction term
        eAtInt: Interval matrix exponential
    """
    
    def __init__(self, A: np.ndarray):
        """
        Constructor for taylorLinSys
        
        Args:
            A: System matrix
        """
        self.A = A
        self.timeStep = None
        self.eAt = None
        self.eAdt = None  # Alias for eAt
        self.Ainv = None
        self.powers = None
        self.error = None
        self.F = None
        self.G = None
        self.V = None
        self.RV = None
        self.Rtrans = None
        self.inputCorr = None
        self.eAtInt = None
        
        # Precompute inverse if it exists
        try:
            if np.linalg.matrix_rank(A) == A.shape[0]:
                self.Ainv = inv(A)
        except:
            self.Ainv = None
    
    def computeField(self, name: str, **kwargs) -> Any:
        """
        Compute and return a field value, computing it if necessary
        
        Args:
            name: Name of the field to compute
            **kwargs: Additional parameters for computation
            
        Returns:
            Computed field value
        """
        if name == 'eAt' or name == 'eAdt':
            return self._computeEAt(**kwargs)
        elif name == 'Ainv':
            return self.Ainv
        elif name == 'Apower':
            return self._computeApower(**kwargs)
        else:
            raise CORAerror('CORA:specialError', f'Unknown field: {name}')
    
    def _computeEAt(self, timeStep: Optional[float] = None) -> np.ndarray:
        """
        Compute matrix exponential e^(A*timeStep)
        
        Args:
            timeStep: Time step size (uses stored value if None)
            
        Returns:
            Matrix exponential
        """
        if timeStep is None:
            timeStep = self.timeStep
        
        if timeStep is None:
            raise CORAerror('CORA:specialError', 'Time step not specified')
        
        # Check if already computed for this time step
        if (self.eAt is not None and self.timeStep is not None and 
            abs(self.timeStep - timeStep) < 1e-12):
            return self.eAt
        
        # Compute matrix exponential
        self.timeStep = timeStep
        self.eAt = expm(self.A * timeStep)
        self.eAdt = self.eAt  # Alias
        
        return self.eAt
    
    def _computeApower(self, ithpower: int) -> np.ndarray:
        """
        Compute i-th power of matrix A
        
        Args:
            ithpower: Power to compute
            
        Returns:
            A^ithpower
        """
        if ithpower == 0:
            return np.eye(self.A.shape[0])
        elif ithpower == 1:
            return self.A
        else:
            # Compute iteratively
            result = self.A
            for i in range(2, ithpower + 1):
                result = result @ self.A
            return result
    
    def getTaylor(self, name: str, **kwargs) -> Any:
        """
        Get Taylor series related values
        
        Args:
            name: Name of the value to get
            **kwargs: Additional parameters
            
        Returns:
            Requested value
        """
        return self.computeField(name, **kwargs) 