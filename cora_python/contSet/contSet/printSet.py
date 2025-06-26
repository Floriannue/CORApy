import numpy as np
from cora_python.g.functions.matlab.validate.preprocessing import setDefaultValues
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
# Import locally to avoid circular imports
# from cora_python.g.functions.verbose.print import printMatrix, printCell, printStruct
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.contSet.contSet import ContSet

def printSet(S, *varargin):
    """
    prints a set such that if one executes this command
    in the workspace, this set would be created
    """
    
    accuracy, do_compact, clear_line = setDefaultValues(['%4.3f', False, True], varargin)
    if isinstance(accuracy, str) and accuracy == 'high':
        accuracy = '%16.16f'
    
    # from cora_python.contSet.contSet.contSet import ContSet
    # from cora_python.matrixSet.matrixSet.matrixSet import matrixSet
    # inputArgsCheck([
    #     [S, 'att', [ContSet, matrixSet]],
    #     [accuracy, 'att', ['char', 'string']],
    #     [do_compact, 'att', ['logical']],
    #     [clear_line, 'att', ['logical']]
    # ])

    abbrev, property_order = S.getPrintSetInfo()

    if do_compact:
        print(f'{type(S).__name__}(', end='')
        for i, pname in enumerate(property_order):
            aux_print_property(getattr(S, pname), accuracy, do_compact)
            if i < len(property_order) - 1:
                print(', ', end='')
        print(')', end='')
    else:
        for pname in property_order:
            print(f'{pname} = ', end='')
            aux_print_property(getattr(S, pname), accuracy, do_compact)
            print(';')
        
        prop_str = ",".join(property_order)
        print(f'{abbrev} = {type(S).__name__}({prop_str});', end='')

    if clear_line:
        print()

def aux_print_property(prop, accuracy, do_compact):
    # Import locally to avoid circular imports
    from cora_python.g.functions.verbose.print import printMatrix, printCell, printStruct
    
    if isinstance(prop, np.ndarray):
        printMatrix(prop, accuracy, do_compact, False)
    elif isinstance(prop, list):
        printCell(prop, accuracy, do_compact, False)
    elif isinstance(prop, dict):
        printStruct(prop, accuracy, do_compact, False)
    elif isinstance(prop, ContSet):
        printSet(prop, accuracy, True, False)
    else:
        raise CORAerror("CORA:noops", prop)

