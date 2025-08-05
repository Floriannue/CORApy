"""
printSpec - prints a specification object such that if one executes this command
   in the workspace, this specification object would be created

Syntax:
    printSpec(spec)
    printSpec(spec, 'high')

Inputs:
    spec - specification
    accuracy - (optional) floating-point precision
    do_compact - (optional) whether to compactly print the set
    clear_line - (optional) whether to finish with '\n'

Outputs:
    -

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 10-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional, Union, List


def printSpec(spec, accuracy: Union[str, float] = '%4.3f', 
              do_compact: bool = False, clear_line: bool = True):
    """
    Print a specification object in a readable format
    
    Args:
        spec: specification object or list of specification objects
        accuracy: floating-point precision format string or 'high' for high precision
        do_compact: whether to compactly print the set
        clear_line: whether to finish with '\n'
    """
    
    # Handle accuracy setting
    if isinstance(accuracy, str) and accuracy == 'high':
        accuracy = '%.16f'
    elif not isinstance(accuracy, str):
        accuracy = f'%.{accuracy}f'
    
    # Ensure spec is a list for uniform handling
    if not isinstance(spec, list):
        specs = [spec]
    else:
        specs = spec
    
    # Print each specification
    for i, spec_obj in enumerate(specs):
        print('Specification(', end='')
        
        if not do_compact:
            print(' ...')
        
        # Print the set
        _print_set(spec_obj.set, accuracy, True, False)
        print(', ', end='')
        
        if not do_compact:
            print('...')
        
        # Print the type
        print(f"'{spec_obj.type}'", end='')
        print(', ', end='')
        
        if not do_compact:
            print('...')
        
        # Print the time
        _print_set(spec_obj.time, accuracy, True, False)
        print(', ', end='')
        
        if not do_compact:
            print('...')
        
        # Print the location
        _print_location(spec_obj.location, True, False)
        
        if not do_compact:
            print(' ...')
        
        print(')', end='')
        
        if clear_line:
            print()


def _print_set(set_obj, accuracy: str, do_compact: bool, clear_line: bool):
    """Print a set object"""
    
    if set_obj is None:
        print('None', end='')
        return
    
    try:
        # Try to use the set's own string representation
        if hasattr(set_obj, '__str__'):
            set_str = str(set_obj)
            # Format numbers in the string if possible
            if hasattr(set_obj, 'center') and hasattr(set_obj, 'generators'):
                # For sets with center and generators
                center = set_obj.center()
                if hasattr(set_obj, 'generators'):
                    gens = set_obj.generators()
                    print(f"{set_obj.__class__.__name__}(", end='')
                    _print_matrix(center, accuracy, True, False)
                    print(', ', end='')
                    _print_matrix(gens, accuracy, True, False)
                    print(')', end='')
                else:
                    print(f"{set_obj.__class__.__name__}(", end='')
                    _print_matrix(center, accuracy, True, False)
                    print(')', end='')
            else:
                print(set_str, end='')
        else:
            print(repr(set_obj), end='')
            
    except Exception as e:
        # Fallback to simple representation
        print(f"{type(set_obj).__name__}(...)", end='')


def _print_location(location, do_compact: bool, clear_line: bool):
    """Print location information"""
    
    if location is None:
        print('None', end='')
        return
    
    if isinstance(location, (list, np.ndarray)):
        _print_matrix(location, '%d', True, False)
    else:
        print(repr(location), end='')


def _print_matrix(matrix, format_str: str, do_compact: bool, clear_line: bool):
    """Print a matrix with specified formatting"""
    
    if matrix is None:
        print('None', end='')
        return
    
    try:
        # Convert to numpy array for uniform handling
        if not isinstance(matrix, np.ndarray):
            if isinstance(matrix, (list, tuple)):
                matrix = np.array(matrix)
            else:
                print(repr(matrix), end='')
                return
        
        if matrix.size == 0:
            print('[]', end='')
            return
        
        # Handle different dimensions
        if matrix.ndim == 0:
            # Scalar
            if 'd' in format_str or 'i' in format_str:
                print(f"{int(matrix)}", end='')
            else:
                print(format_str % matrix, end='')
                
        elif matrix.ndim == 1:
            # Vector
            print('[', end='')
            for i, val in enumerate(matrix):
                if i > 0:
                    print(', ', end='')
                if 'd' in format_str or 'i' in format_str:
                    print(f"{int(val)}", end='')
                else:
                    print(format_str % val, end='')
            print(']', end='')
            
        elif matrix.ndim == 2:
            # Matrix
            print('[', end='')
            for i in range(matrix.shape[0]):
                if i > 0:
                    if do_compact:
                        print('; ', end='')
                    else:
                        print(';\n ', end='')
                
                print('[', end='')
                for j in range(matrix.shape[1]):
                    if j > 0:
                        print(', ', end='')
                    if 'd' in format_str or 'i' in format_str:
                        print(f"{int(matrix[i, j])}", end='')
                    else:
                        print(format_str % matrix[i, j], end='')
                print(']', end='')
            print(']', end='')
        else:
            # Higher dimensions - just show shape
            print(f"array{matrix.shape}", end='')
            
    except Exception as e:
        # Fallback
        print(repr(matrix), end='') 