"""
writeSparseMatrix - write a sparse matrix to file

Syntax:
    empty = writeSparseMatrix(fid,M,var)
    empty = writeSparseMatrix(fid,M,var,taylMod)

Inputs:
    fid - identifier of the file to which the matrix is written
    M - symbolic matrix
    var - name of the matrix that is written
    taylMod - true/false whether inputs are Taylor models

Outputs:
    empty - true if matrix is empty, false otherwise

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: writeMatrix

Authors:       Niklas Kochdumper
Written:       15-July-2017
Last update:   20-July-2017
               24-January-2018
Last revision: 09-October-2024 (MW, unify with writeSparseMatrixTaylorModel)
               Automatic python translation: Florian Nüssel BA 2025
"""

import sympy as sp
import numpy as np
from typing import Any, TextIO
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.string.bracketSubs import bracketSubs
from cora_python.g.functions.verbose.write.writeMatrix import _convert_matlab_indexing_to_python


def writeSparseMatrix(fid: TextIO, M: Any, var: str, *varargin) -> bool:
    """
    Write a sparse matrix to file
    
    Args:
        fid: file identifier (file handle) to which the matrix is written
        M: symbolic matrix (sympy Matrix)
        var: name of the matrix that is written
        *varargin: optional taylMod (bool) - whether inputs are Taylor models
        
    Returns:
        empty: True if matrix is empty, False otherwise
    """
    
    # taylorModel or not?
    # MATLAB: taylMod = setDefaultValues({false},varargin);
    taylMod = setDefaultValues([False], list(varargin))[0]
    
    # MATLAB: [row,col] = find(M~=0);
    # Find non-zero entries
    if isinstance(M, sp.Matrix):
        # For SymPy matrices, check each element individually to correctly identify zeros
        # MATLAB: [row,col] = find(M~=0);
        rows, cols = M.shape
        row_list = []
        col_list = []
        for i in range(rows):
            for j in range(cols):
                elem = M[i, j]
                # Check if element is non-zero (handle symbolic expressions)
                # Skip None values (they shouldn't be in the matrix, but handle gracefully)
                if elem is None:
                    continue
                
                is_nonzero = False
                if isinstance(elem, sp.Basic):
                    # For symbolic expressions, check if it simplifies to zero
                    # First check is_zero property (most reliable)
                    if hasattr(elem, 'is_zero'):
                        if elem.is_zero is True:
                            is_nonzero = False
                        elif elem.is_zero is False:
                            is_nonzero = True
                        else:
                            # is_zero is None - try to simplify and check
                            try:
                                simplified = sp.simplify(elem)
                                # Check if simplified is zero
                                if isinstance(simplified, sp.Basic) and hasattr(simplified, 'is_zero'):
                                    if simplified.is_zero is True:
                                        is_nonzero = False
                                    else:
                                        is_nonzero = True
                                else:
                                    # Not a symbolic expression, check == 0
                                    is_nonzero = (simplified != 0)
                            except:
                                # If simplification fails, check == 0
                                try:
                                    if elem == 0:
                                        is_nonzero = False
                                    else:
                                        is_nonzero = True
                                except:
                                    # If comparison fails, assume non-zero (conservative)
                                    is_nonzero = True
                    else:
                        # No is_zero attribute, check == 0
                        try:
                            if elem == 0:
                                is_nonzero = False
                            else:
                                is_nonzero = True
                        except:
                            # If comparison fails, assume non-zero
                            is_nonzero = True
                else:
                    # Numeric value
                    is_nonzero = (elem != 0 and elem is not None)
                
                if is_nonzero:
                    row_list.append(i + 1)  # MATLAB uses 1-based indexing
                    col_list.append(j + 1)
        row = np.array(row_list)
        col = np.array(col_list)
    else:
        # Assume numpy array or list
        # Convert to numpy if it's a list
        if isinstance(M, list):
            M = np.array(M)
        # Handle 1D and 2D matrices
        if M.ndim == 1:
            # 1D array - treat as column vector
            nonzero_indices = np.nonzero(M != 0)[0]
            row_indices = nonzero_indices
            col_indices = np.zeros_like(nonzero_indices)  # All in column 0
        else:
            # 2D array
            row_indices, col_indices = np.nonzero(M != 0)
        row = row_indices + 1
        col = col_indices + 1
    
    # MATLAB: empty = isempty(row);
    empty = len(row) == 0
    
    # Debug: if verbose, print what we found
    # (This will be removed later, but helps debug the indZero issue)
    
    # loop over all non-zero entries and print them one-by-one
    # MATLAB: if taylMod
    if taylMod:
        # MATLAB: for i=1:length(row)
        for i in range(len(row)):
            # MATLAB: fprintf(fid, '%s(%i,%i) = interval(%s);\n', ...)
            # Get the element value
            if isinstance(M, sp.Matrix):
                elem = M[row[i]-1, col[i]-1]  # Convert to 0-based for sympy
            else:
                elem = M[row[i]-1, col[i]-1]
            
            # Convert to string with bracket substitution
            # MATLAB: bracketSubs(char(vpa(M(row(i),col(i)))))
            # vpa is variable precision arithmetic - in sympy we can use evalf or just convert
            elem_str = str(elem)
            elem_str = bracketSubs(elem_str)
            # Convert MATLAB indexing to Python indexing
            elem_str = _convert_matlab_indexing_to_python(elem_str)
            
            # Write with interval wrapper
            fid.write(f'    {var}[{row[i]-1}, {col[i]-1}] = Interval({elem_str})\n')
    else:
        # MATLAB: for i=1:length(row)
        for i in range(len(row)):
            # MATLAB: fprintf(fid,'%s(%i,%i) = %s;\n',...)
            # Get the element value
            if isinstance(M, sp.Matrix):
                elem = M[row[i]-1, col[i]-1]  # Convert to 0-based for sympy
            else:
                elem = M[row[i]-1, col[i]-1]
            
            # Skip if element is None (shouldn't happen if guard worked correctly)
            if elem is None:
                continue
            
            # Convert to string with bracket substitution
            # MATLAB: bracketSubs(char(M(row(i),col(i))))
            elem_str = str(elem)
            # Skip if string representation is None
            if elem_str == 'None':
                continue
            elem_str = bracketSubs(elem_str)
            # Convert MATLAB indexing to Python indexing
            elem_str = _convert_matlab_indexing_to_python(elem_str)
            
            # Write without interval wrapper
            # Note: Python uses 0-based indexing, but we write 1-based for compatibility
            # Actually, we should write 0-based for Python
            fid.write(f'    {var}[{row[i]-1}, {col[i]-1}] = {elem_str}\n')
    
    return empty

