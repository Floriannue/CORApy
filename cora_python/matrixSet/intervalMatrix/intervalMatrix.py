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
            elif hasattr(matrixCenter, 'C') and hasattr(matrixCenter, 'G'):
                # Convert matZonotope to IntervalMatrix
                # MATLAB: C = matZ.C; D = sum(abs(matZ.G),3);
                from cora_python.contSet.interval.interval import Interval
                C = matrixCenter.C
                D = np.sum(np.abs(matrixCenter.G), axis=2)
                self.int = Interval(C - D, C + D)
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
    def shape(self):
        """Return the shape of the interval matrix"""
        return self.int.inf.shape
    
    @property
    def ndim(self):
        """Return the number of dimensions of the interval matrix (NumPy compatibility)"""
        return self.int.inf.ndim
    
    def __getitem__(self, key):
        """
        Support subscripting for IntervalMatrix (e.g., M[0:2, :])
        
        Args:
            key: slice or tuple of slices
            
        Returns:
            IntervalMatrix with sliced interval
        """
        # Slice the underlying interval
        sliced_int = self.int[key]
        
        # Create new IntervalMatrix from sliced interval
        result = IntervalMatrix.__new__(IntervalMatrix)
        result.int = sliced_int
        return result
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle numpy universal functions
        
        This allows IntervalMatrix to work properly with numpy array operations.
        """
        import numpy as np
        
        if ufunc == np.add:
            if method == '__call__':
                # Handle addition with numpy arrays
                if len(inputs) == 2:
                    if inputs[0] is self:
                        return self.__add__(inputs[1])
                    else:
                        return self.__radd__(inputs[0])
        elif ufunc == np.multiply:
            if method == '__call__':
                # Handle multiplication with numpy arrays
                if len(inputs) == 2:
                    if inputs[0] is self:
                        return self.__mul__(inputs[1])
                    else:
                        return self.__rmul__(inputs[0])
        elif ufunc == np.matmul:
            if method == '__call__':
                # Handle matrix multiplication with numpy arrays
                if len(inputs) == 2:
                    if inputs[0] is self:
                        return self.__matmul__(inputs[1]) if hasattr(self, '__matmul__') else self.__mul__(inputs[1])
                    else:
                        return self.__rmatmul__(inputs[0]) if hasattr(self, '__rmatmul__') else self.__rmul__(inputs[0])
        
        # For other ufuncs, return NotImplemented to let numpy handle it
        return NotImplemented
 