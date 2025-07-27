"""
matZonotope class

Syntax:
    obj = matZonotope()
    obj = matZonotope(C,G)

Inputs:
    C - center matrix (n x m)
    G - h generator matrices stored as (n x m x h)

Outputs:
    obj - generated matZonotope object

Example:
    C = [[0, 0], [0, 0]]
    G = np.zeros((2, 2, 2))
    G[:,:,0] = [[1, 3], [-1, 2]]
    G[:,:,1] = [[2, 0], [1, -1]]
    
    matZ = matZonotope(C,G)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: matrixSet, intervalMatrix, matPolytope

Authors: Matthias Althoff, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 14-September-2006 (MATLAB)
Last update: 25-April-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional, Union, List


class matZonotope:
    """
    matZonotope class for representing matrix zonotopes
    
    A matrix zonotope is defined as:
    M = C + sum_{i=1}^h α_i * G_i
    where C is the center matrix, G_i are generator matrices, and α_i ∈ [-1,1]
    """
    
    def __init__(self, *args):
        """
        Constructor for matZonotope
        
        Args:
            *args: Either no arguments, C only, or (C, G)
        """
        # 0. check number of input arguments
        if len(args) not in [0, 1, 2]:
            raise ValueError(f"matZonotope constructor expects 0, 1, or 2 arguments, got {len(args)}")
        
        # 1. copy constructor
        if len(args) == 1 and isinstance(args[0], matZonotope):
            other = args[0]
            self.C = other.C.copy()
            self.G = other.G.copy()
            return
        
        # 2. parse input arguments
        C, G = self._aux_parseInputArgs(*args)
        
        # 3. check correctness of input arguments
        self._aux_checkInputArgs(C, G)
        
        # 4. update properties
        C, G = self._aux_computeProperties(C, G)
        
        # 5. assign properties
        self.C = C
        self.G = G
    
    def _aux_parseInputArgs(self, *args):
        """Parse input arguments from user and assign to variables"""
        # default values
        G = None
        
        # parse input
        if len(args) == 0:
            C = np.zeros((0, 0))
        elif len(args) == 1:
            C = args[0]
        elif len(args) == 2:
            C, G = args
        
        # fix generators to allow None for no generators
        if G is None:
            G = np.zeros((*C.shape, 0))
        
        return C, G
    
    def _aux_checkInputArgs(self, C, G):
        """Check correctness of input arguments"""
        # Basic input validation
        if not isinstance(C, np.ndarray):
            C = np.asarray(C)
        
        if isinstance(G, list):
            # legacy cell array format
            # check dimensions
            if C.size == 0 and len(G) > 0:
                raise ValueError('Center is empty.')
            
            for i, Gi in enumerate(G):
                # check each generator matrix
                if not np.all(C.shape[:2] == Gi.shape[:2]):
                    raise ValueError(f'Dimension mismatch between center and generator {i}.')
        else:
            if not isinstance(G, np.ndarray):
                G = np.asarray(G)
            
            # check dimensions
            if C.size == 0 and G.size > 0:
                raise ValueError('Center is empty.')
            
            if not np.all(C.shape[:2] == G.shape[:2]):
                raise ValueError('Dimension mismatch between center and generators.')
    
    def _aux_computeProperties(self, C, G):
        """Compute and update properties"""
        if isinstance(G, list):
            # legacy, convert to (n x m x h) shape
            
            # show warning
            import warnings
            warnings.warn('Deprecated constructor for matZonotope using a list of generators. '
                        'Please use a single numeric matrix with dimensions (n x m x h) instead. '
                        'This change was made to improve speed.', UserWarning, stacklevel=2)
            
            # store given generators
            G_legacy = G
            
            # preallocate new generators
            G = np.zeros((*C.shape, len(G_legacy)))
            
            # copy generators
            for i, Gi in enumerate(G_legacy):
                G[:, :, i] = Gi
        
        return C, G
    
    def numgens(self):
        """Get number of generators"""
        return self.G.shape[2] if len(self.G.shape) > 2 else 0
    
    def dim(self):
        """Get dimensions of the matrix zonotope"""
        return self.C.shape
    
    def size(self):
        """Get size of the matrix zonotope"""
        return self.C.shape
    
    def isempty(self):
        """Check if matrix zonotope is empty"""
        return self.C.size == 0
    
    def center(self):
        """Get center matrix"""
        return self.C
    
    def __str__(self):
        """String representation"""
        return f"matZonotope in R^{self.C.shape[0]}x{self.C.shape[1]} with {self.numgens()} generators"
    
    def __repr__(self):
        """Detailed string representation"""
        return self.__str__() 