"""
IntervalMatrix class - Interval matrix representation

Syntax:
    obj = IntervalMatrix()
    obj = IntervalMatrix(C, D)

Inputs:
    C - center matrix
    D - width matrix

Outputs:
    obj - generated object

Example:
   C = np.array([[0, 2], [3, 1]])
   D = np.array([[1, 2], [1, 1]])
   intMat = IntervalMatrix(C, D)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: Interval, MatrixSet, MatZonotope, MatPolytope

Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       18-June-2010 (MATLAB)
Last update:   26-August-2011
               15-June-2016
               06-May-2021
               03-April-2023 (MW, remove properties dim and setting)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, TYPE_CHECKING
from cora_python.matrixSet.matrixSet.matrixSet import MatrixSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.interval.interval import Interval


class IntervalMatrix(MatrixSet):
    """
    IntervalMatrix class for representing interval matrices.
    
    An interval matrix is defined by its center matrix and width matrix,
    representing all matrices within the specified bounds.
    """
    
    def __init__(self, matrixCenter: Optional[Union[np.ndarray, 'IntervalMatrix']] = None, 
                 matrixDelta: Optional[np.ndarray] = None):
        """
        Constructor for IntervalMatrix class.
        
        Args:
            matrixCenter: center matrix or existing IntervalMatrix for copy constructor
            matrixDelta: width matrix (optional)
            
        Raises:
            CORAerror: if no input arguments are provided
        """
        super().__init__()
        
        if matrixCenter is None:
            raise CORAerror('CORA:noInputInSetConstructor', 
                          'IntervalMatrix constructor requires at least one input argument')
        
        if matrixDelta is None:
            if isinstance(matrixCenter, IntervalMatrix):
                # Copy constructor
                self.int = matrixCenter.int
            else:
                # Only center given, radius = 0
                from cora_python.contSet.interval.interval import Interval
                matrixCenter = np.asarray(matrixCenter)
                self.int = Interval(matrixCenter, matrixCenter)
        else:
            # Both center and delta given
            matrixCenter = np.asarray(matrixCenter)
            matrixDelta = np.asarray(matrixDelta)
            
            # Ensure positive matrix deltas
            matrixDelta = np.abs(matrixDelta)
            
            # Create interval from center and delta
            from cora_python.contSet.interval.interval import Interval
            self.int = Interval(matrixCenter - matrixDelta, matrixCenter + matrixDelta)
    
    @property
    def center(self) -> np.ndarray:
        """
        Get the center matrix of the interval matrix.
        
        Returns:
            Center matrix as numpy array
        """
        return self.int.center()
    
    @property
    def delta(self) -> np.ndarray:
        """
        Get the width matrix of the interval matrix.
        
        Returns:
            Width matrix as numpy array
        """
        return self.int.rad()
    
    @property
    def rad(self) -> np.ndarray:
        """
        Get the radius matrix of the interval matrix (alias for delta).
        
        Returns:
            Radius matrix as numpy array
        """
        return self.int.rad()
    
    @property
    def shape(self) -> tuple:
        """
        Get the shape of the interval matrix.
        
        Returns:
            Tuple representing the matrix dimensions
        """
        return self.int.shape
    

    
    def __repr__(self) -> str:
        """String representation of the IntervalMatrix object"""
        return self.display()

    def __str__(self) -> str:
        """String representation of the IntervalMatrix object"""
        return self.display()
    
 