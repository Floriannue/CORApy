import numpy as np
from cora_python.g.functions.matlab.validate.preprocessing import setDefaultValues
from .printMatrix import printMatrix

def printCell(C, *varargin):
    """
    prints a cell array such that if one executes this command
    in the workspace, this cell array would be created
    """
    
    accuracy, do_compact, clear_line = setDefaultValues(['%4.3f', False, True], varargin)
    if isinstance(accuracy, str) and accuracy == 'high':
        accuracy = '%16.16f'
    
    if not C:
        if isinstance(C, list) and len(C) == 0:
            print('[]', end='')
        else:
            print('list()', end='') # Or more specific empty cell repr
        if clear_line:
            print()
        return

    numRows = len(C)
    numCols = len(C[0]) if numRows > 0 and isinstance(C[0], list) else 1
    if not isinstance(C[0], list): # handle 1d list
        C = [C]
        numCols = numRows
        numRows = 1

    print('[', end=' ')
    if numRows > 1 and not do_compact:
        print('...')

    for iRow in range(numRows):
        if numRows > 1:
            print('[', end='')
        for iCol in range(numCols):
            value = C[iRow][iCol]
            
            if isinstance(value, (int, float, np.number)):
                printMatrix(np.array([value]), accuracy, do_compact, False)
            elif isinstance(value, np.ndarray):
                printMatrix(value, accuracy, do_compact, False)
            elif isinstance(value, dict):
                from .printStruct import printStruct
                printStruct(value, accuracy, do_compact, False)
            elif isinstance(value, str):
                print(f"'{value}'", end='')
            elif hasattr(value, 'printSet'):
                value.printSet(accuracy, True, False)
            # elif isinstance(value, 'contDynamics'): # Placeholder
            #     printSystem(value, accuracy, True, False)
            else:
                print(str(value), end='')

            if iCol < numCols - 1 and not do_compact:
                print(',', end='')
            print(' ', end='')

        if numRows > 1:
            print(']', end='')

        if numRows > 1 or iRow < numRows -1:
            if do_compact:
                if iRow < numRows - 1:
                    print(', ', end='')
            else:
                print(', ...')
    
    print(']', end='')
    if clear_line:
        print() 