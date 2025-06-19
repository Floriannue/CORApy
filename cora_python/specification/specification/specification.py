"""
specification - class for specifications for reachability analysis

This class implements specifications that can be used to verify
properties of reachable sets in reachability analysis.

Syntax:
    obj = Specification()
    
    # single set
    obj = Specification(set)
    obj = Specification(set, location)
    obj = Specification(set, type)
    obj = Specification(set, type, location)
    obj = Specification(set, type, time)
    obj = Specification(set, type, location, time)
    obj = Specification(set, type, time, location)
    
    # list of sets
    obj = Specification(list)
    obj = Specification(list, location)
    obj = Specification(list, type)
    obj = Specification(list, type, location)
    obj = Specification(list, type, time)
    obj = Specification(list, type, location, time)
    obj = Specification(list, type, time, location)
    
    # special case: function handle
    obj = Specification(func)
    obj = Specification(func, 'custom')
    obj = Specification(func, 'custom', location)
    obj = Specification(func, 'custom', time)
    obj = Specification(func, 'custom', location, time)
    obj = Specification(func, 'custom', time, location)
    
    # special case: stl formula
    obj = Specification(eq)
    obj = Specification(eq, 'logic')

Inputs:
    set - contSet object that defines the specification
    list - list storing with contSet objects for multiple parallel
           specifications
    type - string that defines the type of specification:
               - 'unsafeSet' (default)
               - 'safeSet'
               - 'invariant' 
               - 'logic'
               - 'custom'
    time - interval defining when the specification is active
    eq - temporal logic formula (class stl)
    func - function handle to a user-provided specification check function
    location - activity of specification in which locations of a HA/pHA

Outputs:
    obj - generated specification object

Example:
    P = Polytope([1, 2], 0)
    spec = Specification(P, 'unsafeSet')

Authors: Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 29-May-2020 (MATLAB)
Last update: 27-November-2022 (MW, add location property and checks) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional, Union, Any, List, Callable
from cora_python.g.functions.matlab.validate.check import assertNarginConstructor
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class Specification:
    """
    Specification class for temporal logic specifications
    
    This class represents specifications that can be verified
    during reachability analysis, such as safety properties,
    invariants, and temporal logic formulas.
    
    Properties:
        set: contSet object that corresponds to the specification
        time: time interval in which the specification is active  
        type: type of specification
        location: location where the specification is active (only hybrid systems)
    """
    
    def __init__(self, *args):
        """
        Constructor for specification objects
        
        Args:
            *args: Variable arguments matching MATLAB constructor patterns
        """
        
        # Initialize properties with defaults
        self.set = None
        self.time = None
        self.type = 'unsafeSet'
        self.location = None
        
                # 0. check number of input arguments
        if len(args) == 0:
            # empty object
            return
        
        assertNarginConstructor([1, 2, 3, 4], len(args))

        # 1. copy constructor
        if len(args) == 1 and isinstance(args[0], Specification):
            obj = args[0]
            self.set = obj.set
            self.time = obj.time
            self.type = obj.type
            self.location = obj.location
            return

        # 2. parse input arguments
        if len(args) >= 1:
            # first input argument: func, eq, set, list
            if callable(args[0]):
                # syntax: obj = Specification(func)
                self.set = args[0]
                self.type = 'custom'
                is_fun_han = True
                is_stl = False
            elif hasattr(args[0], '__class__') and args[0].__class__.__name__ == 'Stl':
                # syntax: obj = Specification(eq)
                self.set = args[0]
                self.type = 'logic'
                is_fun_han = False
                is_stl = True
            elif hasattr(args[0], '__class__') and hasattr(args[0], 'dim'):  # contSet check
                # syntax: obj = Specification(set)
                self.set = args[0]
                is_fun_han = False
                is_stl = False
            elif isinstance(args[0], list):
                # syntax: obj = Specification(list)
                # In Python, constructors cannot return lists
                # This should be handled by a factory function
                raise CORAError('CORA:wrongInputInConstructor',
                    'Cannot create multiple specifications in constructor. '
                    'Use create_specification_list() function instead.')
            else:
                raise CORAError('CORA:wrongInputInConstructor',
                    'First argument must be a contSet, list, function, or STL formula.')

        # ...if list was given, we already returned an array of specification objects

        if len(args) >= 2:
            # second input argument: type, location
            type_, loc, _ = self._read_out_input_arg(args[1], 2)
            
            if type_ is not None:             # type is given
                # ensure correct types for func, eq
                if is_fun_han and type_ != 'custom':
                    raise CORAError('CORA:wrongInputInConstructor',
                        'If the specification is defined using a function handle, '
                        'the property "type" must be "custom".')
                elif is_stl and type_ != 'logic':
                    raise CORAError('CORA:wrongInputInConstructor',
                        'If the specification is defined using an stl formula, '
                        'the property "type" must be "logic".')
                
                # assign value (checked via property validation)
                self.type = type_

            elif loc is not None:          # location is given
                # not supported for stl formulae
                if is_stl:
                    raise CORAError('CORA:notSupported',
                        'Specifications using stl formulae not supported for hybrid systems.')

                # check that format is correct
                self._check_location(loc)

                # assign value
                self.location = loc

        if len(args) >= 3:
            # third input argument: location, time
            _, loc, time = self._read_out_input_arg(args[2], 3)

            # neither time nor location supported for stl formulae
            if is_stl:
                raise CORAError('CORA:notSupported',
                    'Specifications using stl formulae not supported '
                    'for hybrid systems or combined with additional "time" input.')

            if loc is not None:                           # location is given
                # check that format is correct
                self._check_location(loc)

                # assign value
                self.location = loc
            elif time is not None and not self._represents_empty_set(time):  # time is given
                # assign value
                self.time = time

        if len(args) == 4:
            # fourth input argument: location, time
            _, loc, time = self._read_out_input_arg(args[3], 4)

            if loc is not None:          # location is given
                # check that format is correct
                self._check_location(loc)

                # assign value
                self.location = loc
            elif time is not None:     # time is given
                # assign value
                self.time = time

    def _read_out_input_arg(self, arg_in, idx):
        """
        Checks whether given input argument is
        - type (has to be str)
        - location (has to be a list)
        - time (has to be an interval object)
        ...otherwise an error is thrown
        """
        # init as empty
        type_ = None
        loc = None
        time = None

        if isinstance(arg_in, str):
            # Validate type
            valid_types = ['unsafeSet', 'safeSet', 'invariant', 'custom', 'logic']
            if arg_in not in valid_types:
                raise CORAError('CORA:wrongInputInConstructor',
                    f'Invalid type "{arg_in}". Must be one of {valid_types}')
            type_ = arg_in
        elif isinstance(arg_in, (int, float, list, np.ndarray)):
            # numeric for HA, list for pHA
            loc = arg_in
        elif hasattr(arg_in, '__class__') and arg_in.__class__.__name__ == 'Interval':
            time = arg_in
        else:
            # Alter message based on index
            if idx == 2:
                raise CORAError('CORA:wrongInputInConstructor',
                    'The second input argument has to be either a string (type) '
                    'or a numeric array/list (location).')
            elif idx == 3:
                raise CORAError('CORA:wrongInputInConstructor',
                    'The third input argument has to be either a numeric array/list (location) '
                    'or an interval object (time).')
            elif idx == 4:
                raise CORAError('CORA:wrongInputInConstructor',
                    'The fourth input argument has to be either a numeric array/list (location) '
                    'or an interval object (time).')

        return type_, loc, time

    def _check_location(self, loc):
        """
        Checks if the location property has the correct format
        - HA: numeric array or empty, e.g.,
               [1,2] = active in locations 1 and 2
        - pHA: list of numeric arrays, e.g.,
               [[1,2], [1,3]]
               = active in locations 1 and 2 of subcomponent 1,
                    and in locations 1 and 3 of subcomponent 2
        """
        # check whether HA or pHA given
        is_pHA = False
        if isinstance(loc, list) and len(loc) > 0:
            # Check if any element is a list or array (indicating pHA)
            for item in loc:
                if isinstance(item, (list, np.ndarray)):
                    is_pHA = True
                    break
        
        if not isinstance(loc, list) or not is_pHA:
            # HA: vectors/scalar, positive, numeric
            if isinstance(loc, list):
                try:
                    loc = np.array(loc)
                except ValueError:
                    # If can't convert to array, treat as pHA
                    is_pHA = True
            
            if not is_pHA:
                if not isinstance(loc, (int, float, np.ndarray)):
                    raise CORAError('CORA:wrongInputInConstructor',
                        'All entries in the property location have to be '
                        'positive numeric vectors/scalars.')
                
                loc_arr = np.atleast_1d(loc)
                if np.any(np.isnan(loc_arr)) or np.any(np.isinf(loc_arr)) or \
                   not np.all(loc_arr == np.round(loc_arr)) or not np.all(loc_arr > 0):
                    raise CORAError('CORA:wrongInputInConstructor',
                        'All entries in the property location have to be '
                        'positive (non-NaN, non-Inf) numeric vectors/scalars.')
        
        if is_pHA:
            # pHA: each list entry has to be vectors/scalar, positive, numeric
            for i, loc_item in enumerate(loc):
                # Convert to numpy array only if it's a list of numbers
                if isinstance(loc_item, list):
                    try:
                        loc_item = np.array(loc_item)
                    except ValueError:
                        # Handle cases where list elements can't form a regular array
                        pass
                
                if not isinstance(loc_item, (int, float, np.ndarray, list)):
                    raise CORAError('CORA:wrongInputInConstructor',
                        'All entries in the property location have to be '
                        'positive numeric vectors/scalars.')
                
                # Convert to array for validation
                if isinstance(loc_item, list):
                    # For lists that couldn't be converted, check each element
                    for val in loc_item:
                        if not isinstance(val, (int, float)):
                            raise CORAError('CORA:wrongInputInConstructor',
                                'All entries in the property location have to be '
                                'positive numeric vectors/scalars.')
                        if np.isnan(val) or np.isinf(val) or val != int(val) or val <= 0:
                            raise CORAError('CORA:wrongInputInConstructor',
                                'All entries in the property location have to be '
                                'positive (non-NaN, non-Inf) numeric vectors/scalars,\n'
                                '  and must not be larger than the number of sets.')
                else:
                    loc_arr = np.atleast_1d(loc_item)
                    if np.any(np.isnan(loc_arr)) or np.any(np.isinf(loc_arr)) or \
                       not np.all(loc_arr == np.round(loc_arr)) or not np.all(loc_arr > 0):
                        raise CORAError('CORA:wrongInputInConstructor',
                            'All entries in the property location have to be '
                            'positive (non-NaN, non-Inf) numeric vectors/scalars,\n'
                            '  and must not be larger than the number of sets.')

    def _represents_empty_set(self, obj):
        """Check if object represents an empty set"""
        if hasattr(obj, 'representsa_'):
            return obj.representsa_('emptySet', 1e-10)
        return False
    
    def __str__(self) -> str:
        """String representation of specification"""
        if self.time is not None:
            return f"Specification: {self.type} over time {self.time}"
        else:
            return f"Specification: {self.type}"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        """Equality operator"""
        from .eq import eq
        return eq(self, other)
    
    def __ne__(self, other) -> bool:
        """Not equal operator"""
        from .ne import ne
        return ne(self, other)
    
    def isequal(self, other, tol=None) -> bool:
        """Check equality with tolerance"""
        from .isequal import isequal
        return isequal(self, other, tol)
    
    def inverse(self):
        """Invert specification"""
        from .inverse import inverse
        return inverse(self)
    
    def isempty(self) -> bool:
        """Check if specification is empty"""
        from .isempty import isempty
        return isempty(self)
    
    def project(self, dims):
        """Project specification onto subspace"""
        from .project import project
        return project(self, dims)
    
    def check(self, S, *args) -> bool:
        """Check if specification is satisfied"""
        from .check import check
        result, _, _ = check(self, S, *args)
        return result
    
    def add(self, other):
        """Add specifications together"""
        from .add import add
        return add(self, other)
    
    def splitLogic(self):
        """Split into non-logic and logic specifications"""
        from .splitLogic import splitLogic
        return splitLogic(self)
    
    def plot(self, *args, **kwargs):
        """Plot specification"""
        from .plot import plot
        return plot(self, *args, **kwargs)
    
    def plotOverTime(self, *args, **kwargs):
        """Plot specification over time"""
        from .plotOverTime import plotOverTime
        return plotOverTime(self, *args, **kwargs)
    
    def robustness(self, p, *args):
        """Compute robustness of specification"""
        from .robustness import robustness
        return robustness(self, p, *args)
    
    def printSpec(self, *args):
        """Print specification details"""
        from .printSpec import printSpec
        return printSpec(self, *args)


# Convenience functions for creating specifications
def create_safety_specification(safe_set: Any, time: Optional[Any] = None) -> Specification:
    """
    Convenience function to create a safety specification
    
    Args:
        safe_set: Safe set (contSet object)
        time: Time interval (optional)
        
    Returns:
        Specification: Safety specification
    """
    if time is not None:
        return Specification(safe_set, 'safeSet', time)
    else:
        return Specification(safe_set, 'safeSet')


def create_invariant_specification(invariant_set: Any, time: Optional[Any] = None) -> Specification:
    """
    Convenience function to create an invariant specification
    
    Args:
        invariant_set: Invariant set (contSet object)
        time: Time interval (optional)
        
    Returns:
        Specification: Invariant specification
    """
    if time is not None:
        return Specification(invariant_set, 'invariant', time)
    else:
        return Specification(invariant_set, 'invariant')


def create_unsafe_specification(unsafe_set: Any, time: Optional[Any] = None) -> Specification:
    """
    Create an unsafe specification (convenience function)
    
    Args:
        unsafe_set: Set representing unsafe region
        time: Optional time interval
        
    Returns:
        Specification object with type 'unsafeSet'
    """
    if time is not None:
        return Specification(unsafe_set, 'unsafeSet', time)
    else:
        return Specification(unsafe_set, 'unsafeSet')


def create_specification_list(sets_list: list, spec_type: str = 'unsafeSet', 
                             time: Optional[Any] = None, location: Optional[Any] = None) -> list:
    """
    Create a list of specification objects from a list of sets
    
    This function mimics the MATLAB behavior where passing a list to the 
    specification constructor creates multiple specification objects.
    
    Args:
        sets_list: List of contSet objects, function handles, or STL formulas
        spec_type: Type of specification ('unsafeSet', 'safeSet', 'invariant', 'custom', 'logic')
        time: Optional time interval to apply to all specifications
        location: Optional location constraint to apply to all specifications
        
    Returns:
        List of Specification objects
        
    Examples:
        >>> sets = [zonotope(...), interval(...), polytope(...)]
        >>> specs = create_specification_list(sets, 'safeSet')
        >>> len(specs)
        3
    """
    # Validate input list
    if not isinstance(sets_list, list) or len(sets_list) == 0:
        raise CORAError('CORA:wrongInputInConstructor',
            'Input must be a non-empty list.')
    
    # Check what types of objects we have
    are_contsets = all(hasattr(x, '__class__') and hasattr(x, 'dim') for x in sets_list)
    are_stl = all(hasattr(x, '__class__') and x.__class__.__name__ == 'Stl' for x in sets_list)
    are_fun_han = all(callable(x) for x in sets_list)
    
    if not (are_contsets or are_stl or are_fun_han):
        raise CORAError('CORA:wrongInputInConstructor',
            'All items in the list must be of the same type: '
            'contSet objects, STL formulas, or function handles.')
    
    # Determine the appropriate type if not explicitly provided
    if are_stl and spec_type != 'logic':
        spec_type = 'logic'
    elif are_fun_han and spec_type != 'custom':
        spec_type = 'custom'
    
    # Create specifications
    specs = []
    for item in sets_list:
        # Create basic specification
        if time is not None and location is not None:
            spec = Specification(item, spec_type, time, location)
        elif time is not None:
            spec = Specification(item, spec_type, time)
        elif location is not None:
            spec = Specification(item, spec_type, location)
        else:
            spec = Specification(item, spec_type)
        
        specs.append(spec)
    
    return specs 