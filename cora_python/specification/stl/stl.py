"""
stl - class representing a Signal Temporal Logic (STL) formula

TRANSLATED FROM: cora_matlab/specification/@stl/stl.m

Syntax:
    obj = stl(name)
    obj = stl(name,num)
    obj = stl(true)
    obj = stl(false)

Inputs:
    name - name of the STL-variable
    num - dimension of the variable

Outputs:
    obj - generated stl variable

Example:
    x = stl('x',2);
    eq = until(x(1) <= 5,x(2) > 3 & x(1) <= 2, interval(0.1,0.2))

Authors:       Niklas Kochdumper, Benedikt Seidl
Written:       09-November-2022
Last update:   07-February-2024 (FL, replace from and to by interval)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Optional, List, Union, Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class Stl:
    """
    stl - class representing a Signal Temporal Logic (STL) formula
    
    Properties:
        type: type of the operator ('until', '&', etc.)
        lhs: object forming left hand side of operator
        rhs: object forming right hand side of operator
        interval: time interval of the formula
        variables: list of variables present in the formula
        var: name of the current variable
        temporal: temporal (true) or non-temporal (false)
        logic: propositional formula (true) or not (false)
        id: unique identifier for the formula
    """
    
    def __init__(self, name: Union[str, bool], num: Optional[int] = None):
        """
        Constructor for stl class
        
        Args:
            name: name of the STL-variable or boolean value (True/False)
            num: dimension of the variable (optional)
        """
        # Catch the case where the input is a logic value
        if isinstance(name, bool):
            if name:
                self.type = 'true'
            else:
                self.type = 'false'
            self.variables = []
            self.logic = True
            self.lhs = None
            self.rhs = None
            self.interval = None
            self.var = None
            self.temporal = False
            self.id = None
            return
        
        # Parse and check input arguments
        if not isinstance(name, str):
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Input "name" has to be a string!')
        
        # Construct variable names
        if num is not None:
            if not isinstance(num, int) or num <= 0:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Input "num" has to be a positive integer!')
            self.variables = [f'{name}{i+1}' for i in range(num)]
        else:
            self.variables = [name]
        
        # Initialize properties
        self.type = 'variable'
        self.var = name
        self.lhs = None
        self.rhs = None
        self.interval = None
        self.temporal = False
        self.logic = False
        self.id = None
    
    def __getitem__(self, index: int) -> 'Stl':
        """
        Access individual variable components (e.g., x[0] for x(1) in MATLAB)
        Note: Python uses 0-based indexing
        """
        if self.type != 'variable':
            raise CORAerror('CORA:notSupported',
                          'Indexing is only supported for variable type stl objects!')
        
        if not isinstance(index, int) or index < 0 or index >= len(self.variables):
            raise CORAerror('CORA:wrongValue',
                          f'Index {index} is out of range for {len(self.variables)} variables!')
        
        # Create a new stl object for the indexed variable
        res = Stl(self.variables[index])
        res.type = 'variable'
        res.var = self.variables[index]
        return res
    
    def __getattr__(self, name: str):
        """
        Handle method name aliases for reserved keywords
        This allows x.finally() and x.in() to work as aliases for x.finally_() and x.in_()
        """
        if name == 'finally':
            return self.finally_
        elif name == 'in':
            return self.in_
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    # Methods in_ and finally_ are defined in separate files and attached in __init__.py

