"""
writeSparseMatrixOptimized - writes sparse matrix using optimized function output

Syntax:
    empty = writeSparseMatrixOptimized(fid,ind,var,tayMod)

Inputs:
    fid - identifier of the file to which the matrix is written
    ind - struct with fields .row, .col, .index containing indices
    var - name of the matrix that is written
    tayMod - true/false whether inputs are Taylor models

Outputs:
    empty - true/false whether index is empty

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: writeSparseMatrix, writeMatrix

Authors:       ???
Written:       ---
Last update:   09-October-2024 (MW, add output argument, sprintf)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, TextIO, Dict


def writeSparseMatrixOptimized(fid: TextIO, ind: Dict[str, Any], var: str, tayMod: bool) -> bool:
    """
    Writes sparse matrix using optimized function output
    
    Args:
        fid: file identifier (file handle) to which the matrix is written
        ind: dict with keys 'row', 'col', 'index' containing indices
        var: name of the matrix that is written
        tayMod: True/False whether inputs are Taylor models
        
    Returns:
        empty: True if index is empty, False otherwise
    """
    
    # variable
    # MATLAB: if contains(var,'Hf')
    if 'Hf' in var:
        # MATLAB: out = 'outDyn';
        out = 'outDyn'
    # MATLAB: elseif contains(var,'Hg')
    elif 'Hg' in var:
        # MATLAB: out = 'outCon';
        out = 'outCon'
    else:  # for third-order tensors
        # MATLAB: out = 'out';
        out = 'out'
    
    # MATLAB: empty = isempty(ind.row);
    empty = len(ind.get('row', [])) == 0
    
    # loop over all non-empty entries
    # MATLAB: for i=1:length(ind.row)
    for i in range(len(ind.get('row', []))):
        # MATLAB: if tayMod
        if tayMod:
            # MATLAB: str = sprintf('%s(%i,%i) = interval(%s(%i));',...)
            # Python uses 0-based indexing
            str_line = f'    {var}[{ind["row"][i]-1}, {ind["col"][i]-1}] = Interval({out}[{ind["index"][i]-1}])\n'
        else:
            # MATLAB: str = sprintf('%s(%i,%i) = %s(%i);',...)
            str_line = f'    {var}[{ind["row"][i]-1}, {ind["col"][i]-1}] = {out}[{ind["index"][i]-1}]\n'
        
        # MATLAB: fprintf(fid, '%s\n', str);
        fid.write(str_line)
    
    return empty

