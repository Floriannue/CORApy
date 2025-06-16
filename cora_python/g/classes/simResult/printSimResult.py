"""
printSimResult - prints a simResult object such that if one executes this command
    in the workspace, this simResult object would be created

Syntax:
    printSimResult(simResult)
    printSimResult(simResult,'high')

Inputs:
    simRes - simResult object
    accuracy - (optional) floating-point precision
    doCompact - (optional) whether to compactly print the set
    clearLine - (optional) whether to finish with '\n'

Outputs:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: specification
"""

from typing import TYPE_CHECKING, Optional, Union
import numpy as np

if TYPE_CHECKING:
    from .simResult import SimResult

def printSimResult(simRes: 'SimResult', 
                  accuracy: Union[str, float] = '%4.3f%s',
                  doCompact: bool = False,
                  clearLine: bool = True) -> None:
    """
    Prints a simResult object in a readable format.
    
    Args:
        simRes: simResult object
        accuracy: floating-point precision format string or 'high' for high precision
        doCompact: whether to compactly print the set
        clearLine: whether to finish with newline
    """
    # Handle accuracy parameter
    if isinstance(accuracy, str) and accuracy == 'high':
        accuracy = '%.16f'
    elif isinstance(accuracy, str):
        # Remove the %s part if present for Python formatting
        accuracy = accuracy.replace('%s', '')
    elif isinstance(accuracy, float):
        accuracy = f'%.{int(accuracy)}f'
    else:
        accuracy = '%.3f'
    
    # Start constructor call
    print('SimResult(', end='')
    if not doCompact:
        print(' ...')
    
    # Print x (states)
    _print_cell_array(simRes.x, accuracy, True, False)
    print(', ', end='')
    if not doCompact:
        print('...')
    
    # Print t (time)
    _print_cell_array(simRes.t, accuracy, True, False)
    print(', ', end='')
    if not doCompact:
        print('...')
    
    # Print loc (location)
    loc = simRes.loc
    if isinstance(loc, list):
        loc = np.array(loc)
    _print_matrix(loc, '%d', True, False)
    print(', ', end='')
    if not doCompact:
        print('...')
    
    # Print y (outputs)
    _print_cell_array(simRes.y, accuracy, True, False)
    print(', ', end='')
    if not doCompact:
        print('...')
    
    # Print a (algebraic variables)
    _print_cell_array(simRes.a, accuracy, True, False)
    if not doCompact:
        print(' ...')
    
    print(')', end='')
    
    if clearLine:
        print()


def _print_cell_array(cell_array: list, accuracy: str, 
                     is_first: bool = True, clear_line: bool = True) -> None:
    """Print a cell array (list) in MATLAB-like format"""
    if not cell_array:
        print('[]', end='')
        return
    
    print('[', end='')
    for i, item in enumerate(cell_array):
        if i > 0:
            print(', ', end='')
        
        if isinstance(item, np.ndarray):
            _print_matrix(item, accuracy, i == 0, False)
        elif isinstance(item, (list, tuple)):
            _print_matrix(np.array(item), accuracy, i == 0, False)
        else:
            print(f'{item:{accuracy}}', end='')
    
    print(']', end='')


def _print_matrix(matrix: np.ndarray, accuracy: str, 
                 is_first: bool = True, clear_line: bool = True) -> None:
    """Print a matrix in MATLAB-like format"""
    if matrix.size == 0:
        print('[]', end='')
        return
    
    if matrix.ndim == 0:
        # Scalar
        print(f'{matrix.item():{accuracy}}', end='')
        return
    elif matrix.ndim == 1:
        # Vector
        print('[', end='')
        for i, val in enumerate(matrix):
            if i > 0:
                print(', ', end='')
            print(f'{val:{accuracy}}', end='')
        print(']', end='')
    else:
        # Matrix
        print('[', end='')
        for i, row in enumerate(matrix):
            if i > 0:
                print('; ', end='')
            for j, val in enumerate(row):
                if j > 0:
                    print(', ', end='')
                print(f'{val:{accuracy}}', end='')
        print(']', end='') 