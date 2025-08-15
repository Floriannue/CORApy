"""
matPolytope class

This class represents matrix polytopes, defined by a set of vertices.

Syntax:
    obj = matPolytope(V)

Inputs:
    V - numpy array storing the vertices (n x m x N), where N is the number of vertices.
        Each (n x m) slice represents a vertex (a matrix).

Outputs:
    obj - generated matPolytope object

Authors:       Matthias Althoff, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       21-June-2010 (MATLAB)
Last update:   02-May-2024 (TL, new structure of V) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, List, Tuple, Any
from cora_python.matrixSet.matrixSet.matrixSet import MatrixSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED

class MatPolytope(MatrixSet):
    """
    Class for representing matrix polytopes.
    
    A matrix polytope is defined by a set of vertices, where each vertex is a matrix.
    """

    # Properties
    V: np.ndarray # vertices (n x m x N)

    def __init__(self, V: np.ndarray = None):
        """
        Constructor for matPolytope objects.

        Args:
            V: NumPy array storing the vertices (n x m x N).
        """
        super().__init__() # Call parent constructor

        # 0. check number of input arguments (mimic MATLAB's nargin behavior)
        # Python's default arguments make nargin check less direct.
        # This assert applies to external calls, internal calls should ensure proper V.
        # We'll handle V=None as 0 arguments, V as 1 argument.
        num_args_in = 0 if V is None else 1
        assertNarginConstructor([0, 1], num_args_in) # Supports 0 or 1 input argument

        # 1. handle empty input (MATLAB: nargin == 0)
        if V is None:
            V = np.array([]).reshape(0,0,0) # Default to empty 3D array for consistency

        # 2. check input arguments (calls auxiliary function)
        _aux_checkInputArgs(V)

        # 3. compute properties (calls auxiliary function)
        self.V = _aux_computeProperties(V)


# Auxiliary functions -----------------------------------------------------

def _aux_checkInputArgs(V: np.ndarray) -> None:
    """Check correctness of input arguments for matPolytope constructor."""

    # only check if macro set to true
    if CHECKS_ENABLED(): # Call CHECKS_ENABLED as a function

        # In MATLAB, iscell(V) was used for legacy input. In Python, we will check if V
        # is a list (mimicking cell array behavior for testing or compatibility) or a numpy array.
        # However, the problem description states V is a numpy array (n x m x N), so we prioritize that.
        
        if not isinstance(V, np.ndarray):
            # Handle cases where V might be a list of matrices if we supported legacy input
            # For now, we strictly expect numpy array for V.
            raise CORAerror('CORA:wrongInputInConstructor', 'Input vertices V must be a NumPy array.')

        if V.ndim != 3 and V.size > 0:
             raise CORAerror('CORA:wrongInputInConstructor', 'Input vertices V must be a 3D NumPy array (n x m x N).')

        if np.any(np.isnan(V)):
            raise CORAerror('CORA:wrongInputInConstructor', 'Input vertices V must not contain NaN values.')

def _aux_computeProperties(V: np.ndarray) -> np.ndarray:
    """Compute properties and normalize vertices V."""

    # MATLAB's aux_computeProperties handles legacy cell array input and empty input.
    # For Python, we assume V is already a NumPy array due to _aux_checkInputArgs.
    # The main task here is to correctly initialize empty V.

    if V.size == 0:
        # MATLAB: if isempty(V), V = zeros([size(V,1:2),1]);
        # This ensures that even empty V has correct matrix dimensions if n,m are known
        # For matPolytope, V is n x m x N. If V is truly empty (e.g., np.array([])), its shape is (0,)
        # We need to reshape it to (0,0,0) to maintain consistency with the 3D structure.
        # If V was passed as np.array([]).reshape(n,m,0) from reshape in polytope/matPolytope.py,
        # then its shape would be (n,m,0). If it's a completely empty array from constructor,
        # then it's (0,). Handle both.
        if V.ndim == 3: # If it's already a 3D empty array from reshape
            return V # Already in desired format (n,m,0)
        else:
            # If V is a flat empty array, convert to 0x0x0 for consistency with 3D empty arrays
            return np.array([]).reshape(0,0,0)
    
    return V
