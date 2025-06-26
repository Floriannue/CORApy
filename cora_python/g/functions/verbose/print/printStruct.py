# placeholder

import numpy as np
from cora_python.g.functions.matlab.validate.preprocessing import setDefaultValues
from .printMatrix import printMatrix

def printStruct(S, *varargin):
    """
    prints a struct such that if one executes this command
    in the workspace, this struct would be created
    """
    if not isinstance(S, dict):
        # In python, we print dicts, not general objects as "structs"
        # For other objects, rely on their __str__ or __repr__
        print(S)
        return

    accuracy, do_compact, clear_line = setDefaultValues(['%4.3f', False, True], varargin)
    if isinstance(accuracy, str) and accuracy == 'high':
        accuracy = '%16.16f'

    print('dict(', end='')

    names = list(S.keys())
    numNames = len(names)

    if numNames > 1 and not do_compact:
        print('...')

    for i, name in enumerate(names):
        if numNames > 1 and not do_compact:
            print('    ', end='')
        
        print(f"'{name}'=", end='')

        value = S[name]
        if isinstance(value, (int, float, np.number, np.ndarray)):
            printMatrix(np.array(value), accuracy, do_compact, False)
        elif isinstance(value, dict):
            printStruct(value, accuracy, do_compact, False)
        elif isinstance(value, list):
            print('[', end='')
            from .printCell import printCell
            printCell([value], accuracy, do_compact, False)
            print(']', end='')
        elif isinstance(value, str):
            print(f"'{value}'", end='')
        elif hasattr(value, 'printSet'):
            value.printSet(accuracy, True, False)
        # elif isinstance(value, 'contDynamics'):
        #     printSystem(value, accuracy, True, False)
        else:
            print(str(value), end='')

        if i < numNames - 1:
            print(',', end='')

        if numNames > 1 and not do_compact:
            print('...')
    
    print(')', end='')

    if clear_line:
        print()
