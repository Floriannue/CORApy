import numpy as np
from scipy.sparse import issparse, find

from cora_python.g.functions.matlab.validate.preprocessing import setDefaultValues
# from cora_python.g.functions.matlab.validate.check import inputArgsCheck

def printMatrix(M, *varargin):
    """
    prints an matrix such that if one executes this command
    in the workspace, this matrix would be created
    """

    defaults = setDefaultValues(['%4.3f', False, True], varargin)
    accuracy, do_compact, clear_line = defaults
    if isinstance(accuracy, str) and accuracy == 'high':
        accuracy = '%16.16f'
    
    # inputArgsCheck([
    #     [M, 'att', ['numeric']],
    #     [accuracy, 'att', ['char', 'string']],
    #     [do_compact, 'att', ['logical']],
    #     [clear_line, 'att', ['logical']]
    # ])

    if not isinstance(M, np.ndarray):
        M = np.array(M)

    if M.size == 0:
        if all(dim == 0 for dim in M.shape):
            print('[]', end='')
        else:
            print(f'np.zeros(({",".join(map(str, M.shape))}))', end='')
        if clear_line:
            print()
        return

    if M.size == 1:
        item = M.item()
        if np.isinf(item) or np.isnan(item):
            print(str(item), end='')
        elif round(item) == item:
            print(f'{int(item)}', end='')
        else:
            print(accuracy % item, end='')
        if clear_line:
            print()
        return

    if np.unique(M).size == 1:
        n_rows, n_clmns = M.shape
        val = M.flat[0]
        if val == 0:
            mat_string = f'np.zeros(({n_rows},{n_clmns}))'
        else:
            format_string = f"({accuracy} * np.ones(({n_rows},{n_clmns})))"
            mat_string = format_string % val
        
        if issparse(M):
            print(f"sparse({mat_string})", end='')
        else:
            print(mat_string, end='')
        if clear_line:
            print()
        return

    if issparse(M):
        i, j, v = find(M)
        print('sparse((', end='')
        if not do_compact: print(' ...\n', end='')
        printMatrix(i, '%i', True, False)
        print(', ', end='')
        if not do_compact: print('...\n', end='')
        printMatrix(j, '%i', True, False)
        print(', ', end='')
        if not do_compact: print('...\n', end='')
        printMatrix(v, accuracy, True, False)
        if not do_compact: print(' ...\n', end='')
        print('))', end='')
        if clear_line:
            print()
        return

    if M.ndim > 2:
        print('np.reshape(', end='')
        printMatrix(M.flatten(), accuracy, True, False)
        shape_str = ",".join(map(str, M.shape))
        print(f', ({shape_str}))', end='')
        if clear_line:
            print()
        return

    numRows, numCols = M.shape
    print('np.array([', end='')
    if numRows > 1 and not do_compact:
        print('...')
        
    for iRow in range(numRows):
        if numRows > 1:
            print('[', end='')

        for iCol in range(numCols):
            print(accuracy % M[iRow, iCol], end='')
            if iCol < numCols - 1:
                print(', ' if do_compact else ',', end='')
        
        if numRows > 1:
            print(']', end='')

        if iRow < numRows - 1:
            print(', ' if do_compact else ', ...')
    
    print('])', end='')
    if clear_line:
        print() 