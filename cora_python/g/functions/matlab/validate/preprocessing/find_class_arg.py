from typing import Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def find_class_arg(obj1: Any, obj2: Any, classname: str) -> tuple[Any, Any]:
    """
    Finds the object with the specified class name.

    This function checks which of the two input objects, obj1 or obj2, is an
    instance of the class specified by 'classname'. It returns the object that
    is an instance of the class as the first value, and the other object as
    the second value. If neither object is an instance of the specified
    class, it raises a CORAerror.

    Args:
        obj1: The first object.
        obj2: The second object.
        classname: The name of the class to check for.

    Returns:
        A tuple containing the found object and the other object.

    Raises:
        CORAerror: If neither of the objects is an instance of the given class.
    """
    if obj1.__class__.__name__ == classname:
        return obj1, obj2
    elif obj2.__class__.__name__ == classname:
        return obj2, obj1
    else:
        for base_class in obj1.__class__.__mro__:
            if base_class.__name__ == classname:
                return obj1, obj2
        for base_class in obj2.__class__.__mro__:
            if base_class.__name__ == classname:
                return obj2, obj1

        raise CORAerror('CORA:wrongValue', 'first/second', classname) 