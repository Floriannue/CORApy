"""
find_class_arg - finds the obj with specified class name, or throws error

Syntax:
   [obj,S] = find_class_arg(obj1,obj2,classname)

Inputs:
   obj1      - first object
   obj2      - second object
   classname - string specifying the classname to find (e.g., 'ellipsoid', 'zonotope')

Outputs:
   obj - the object of the specified classname
   S   - the other object

Authors:       (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       ---
Last update:   ---
Last revision: ---
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def find_class_arg(obj1, obj2, classname: str):

    is_obj1_target = False
    is_obj2_target = False

    # Check if obj1 is an instance of classname
    # We need to handle potential circular imports for class objects. Direct isinstance might require the class to be imported.
    # Using __class__.__name__.lower() for robustness, as classname string from MATLAB is usually lowercase.
    # For numeric types, classname will be 'numeric' which is not a Python class name directly.
    if isinstance(obj1, np.ndarray) and classname.lower() == 'numeric':
        is_obj1_target = True
    elif hasattr(obj1, '__class__') and obj1.__class__.__name__.lower() == classname.lower():
        is_obj1_target = True

    # Check if obj2 is an instance of classname
    if isinstance(obj2, np.ndarray) and classname.lower() == 'numeric':
        is_obj2_target = True
    elif hasattr(obj2, '__class__') and obj2.__class__.__name__.lower() == classname.lower():
        is_obj2_target = True

    if is_obj1_target and not is_obj2_target:
        return obj1, obj2
    elif not is_obj1_target and is_obj2_target:
        return obj2, obj1
    elif is_obj1_target and is_obj2_target:
        raise CORAerror('CORA:wrongValue', f'Expected only one argument to be of class {classname}, but both are.')
    else:
        raise CORAerror('CORA:wrongValue', f'Neither argument is of class {classname}.') 