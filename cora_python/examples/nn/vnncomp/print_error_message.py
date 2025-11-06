"""
print_error_message - formatted printing of an error message

Syntax:
    print_error_message(e)

Inputs:
    e - Exception object

Outputs:
    None

References:
    [1] VNN-COMP'24

Authors:       Lukas Koller
Written:       11-August-2025
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import os
import sys
import traceback
from typing import Any


def print_error_message(e: Exception):
    """
    Print formatted error message with stack trace.
    
    Args:
        e: Exception object
    """
    # Print the error message
    print()
    print(f'Unexpected Error! --- {str(e)}')
    
    # Get the traceback
    tb = traceback.extract_tb(sys.exc_info()[2]) if sys.exc_info()[2] else []
    
    # Print the stack
    for frame in tb:
        func_name, class_name = extract_stack_info(frame)
        print(f' --- {class_name}/{func_name} [{frame.lineno}]')
    
    print()


def extract_stack_info(frame: traceback.FrameSummary) -> tuple:
    """
    Extract function and class names from a stack frame.
    
    Args:
        frame: A FrameSummary object from traceback
        
    Returns:
        Tuple of (func_name, class_name)
    """
    # Get the function name
    func_name = frame.name
    
    # Get the file path
    file_path = frame.filename
    
    # Extract directory and filename
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(base_name)[0]
    
    # Check if the function is in a class directory (Python doesn't use @ like MATLAB)
    # In Python, we can check if it's a method by looking for 'self' or 'cls' in the code
    # For simplicity, we'll use the filename as the class name
    class_name = file_name_no_ext
    
    # Handle methods vs functions
    # If the function name is different from the filename, it might be a nested function
    if func_name != file_name_no_ext and func_name != '<module>':
        # Keep function name as is
        pass
    elif func_name == '<module>':
        # Top-level code
        func_name = file_name_no_ext
    
    return func_name, class_name


if __name__ == '__main__':
    # Test the error printing
    def test_function():
        try:
            # Simulate an error
            x = 1 / 0
        except Exception as e:
            print_error_message(e)
    
    test_function()

