"""
CORAerror - central hub for all error messages thrown by CORA functions

This module provides the CORAerror exception class that mimics MATLAB's CORAerror
functionality for centralized error handling in CORA.

Syntax:
    raise CORAerror(identifier, message)

Inputs:
    identifier - name of CORA error (e.g., 'CORA:wrongInputInConstructor')
    message - additional information about the error

Authors: Mingrui Wang, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 05-April-2022 (MATLAB)
Python translation: 2025
"""

import inspect
from typing import Optional, Any


class CORAerror(Exception):
    """
    Custom exception class for CORA errors
    
    This class provides centralized error handling for all CORA functions,
    similar to MATLAB's CORAerror function.
    
    Attributes:
        identifier: Error identifier (e.g., 'CORA:wrongInputInConstructor')
        message: Additional error message
        filename: Name of the file where error occurred
        classname: Name of the class where error occurred
        functionname: Name of the function where error occurred
    """
    
    def __init__(self, identifier: str, message: str = "", *args):
        """
        Initialize CORAerror
        
        Args:
            identifier: Error identifier string
            message: Additional error message
            *args: Additional arguments for specific error types
        """
        self.identifier = identifier
        self.message = message
        self.args_list = list(args)
        
        # Get caller information
        self.filename, self.classname, self.functionname = self._get_caller_info()
        
        # Generate error message based on identifier
        full_message = self._generate_error_message()
        
        super().__init__(full_message)
    
    def _get_caller_info(self):
        """Get information about the calling function"""
        try:
            # Get the call stack
            stack = inspect.stack()
            
            # Find the first frame that's not in this file
            for frame_info in stack[1:]:
                if 'CORAerror.py' not in frame_info.filename:
                    filename = frame_info.filename.split('/')[-1].split('\\')[-1]
                    if filename.endswith('.py'):
                        filename = filename[:-3]
                    
                    functionname = frame_info.function
                    
                    # Try to determine class name from the frame
                    classname = functionname
                    if 'self' in frame_info.frame.f_locals:
                        classname = frame_info.frame.f_locals['self'].__class__.__name__
                    
                    return filename, classname, functionname
            
            return 'unknown', 'unknown', 'unknown'
        except:
            return 'unknown', 'unknown', 'unknown'
    
    def _generate_error_message(self) -> str:
        """Generate error message based on identifier"""
        
        # Standard help message
        if self.classname != self.functionname:
            helpmsg = f"  Type 'help {self.classname}.{self.functionname}' for more information."
        else:
            helpmsg = f"  Type 'help {self.classname}' for more information."
        
        # Generate error message based on identifier
        if self.identifier == 'CORA:wrongInputInConstructor':
            return f"Wrong input arguments for constructor of class: {self.classname}\n  {self.message}\n{helpmsg}"
        
        elif self.identifier == 'CORA:noInputInSetConstructor':
            infomsg = f'Please consider calling {self.classname}.empty or {self.classname}.Inf instead.'
            return f"No input arguments for constructor of class: {self.classname}\n  {infomsg}\n{helpmsg}"
        
        elif self.identifier == 'CORA:dimensionMismatch':
            if len(self.args_list) >= 2:
                obj1, obj2 = self.args_list[0], self.args_list[1]
                name1, name2 = type(obj1).__name__, type(obj2).__name__
                
                # Get dimensions or sizes
                if hasattr(obj1, 'dim') and callable(obj1.dim):
                    dim1 = obj1.dim()
                else:
                    dim1 = getattr(obj1, 'shape', 'unknown')
                
                if hasattr(obj2, 'dim') and callable(obj2.dim):
                    dim2 = obj2.dim()
                else:
                    dim2 = getattr(obj2, 'shape', 'unknown')
                
                return f"The first object ({name1}) has dimension/size {dim1}, but the second object ({name2}) has dimension/size {dim2}."
            else:
                return f"Dimension mismatch between objects. {self.message}"
        
        elif self.identifier == 'CORA:emptySet':
            return 'Set is empty!'
        
        elif self.identifier == 'CORA:fileNotFound':
            return f'File with name {self.message} could not be found.'
        
        elif self.identifier == 'CORA:wrongValue':
            if len(self.args_list) >= 1:
                explains = self.args_list[0] if self.args_list else self.message
                if 'name-value pair' in self.message:
                    return f"Wrong value for {self.message}.\n  The right value: {explains}\n{helpmsg}"
                else:
                    return f"Wrong value for the {self.message} input argument.\n  The right value: {explains}\n{helpmsg}"
            else:
                return f"Wrong value: {self.message}\n{helpmsg}"
        
        elif self.identifier == 'CORA:plotProperties':
            if self.message:
                return self.message
            else:
                return 'Incorrect plotting properties specified.'
        
        elif self.identifier == 'CORA:notSupported':
            return self.message
        
        elif self.identifier == 'CORA:notDefined':
            return f'Undefined functionality: {self.message}'
        
        elif self.identifier == 'CORA:specialError':
            return self.message
        
        elif self.identifier == 'CORA:noops':
            if self.message:
                return self.message
            elif self.args_list:
                classlist = ", ".join([type(arg).__name__ for arg in self.args_list])
                return f"The function '{self.functionname}' is not implemented for the following arguments:\n  {classlist}.\n{helpmsg}"
        
        elif self.identifier == 'CORA:noExactAlg':
            if self.args_list:
                classlist = ", ".join([type(arg).__name__ for arg in self.args_list])
                return f"There is no exact algorithm for function {self.functionname} with input arguments:\n  {classlist}"
            else:
                return f"There is no exact algorithm for function {self.functionname}."
        
        elif self.identifier == 'CORA:solverIssue':
            solver = f" ({self.message})" if self.message else ""
            return f"Solver{solver} in {self.functionname} failed due to numerical/other issues!\n{helpmsg}"
        
        elif self.identifier == 'CORA:outOfDomain':
            return f"Input is not inside the valid domain (function {self.functionname}).\n{self.message}\n{helpmsg}"
        
        else:
            # Default case for unknown identifiers
            return f"{self.identifier}: {self.message}"

