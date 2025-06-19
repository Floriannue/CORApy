"""
copy - copies the contSet object (used for dynamic dispatch)

This function creates a copy of a contSet object. It should be overridden
in subclasses to provide specific copying logic.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2024 (MATLAB)
Python translation: 2025
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def copy(S: 'ContSet') -> 'ContSet':
    """
    Copies the contSet object (used for dynamic dispatch).

    This is the base implementation that throws an error. Subclasses must
    override this method by providing a `copy.py` file in their own
    module to provide specific copying logic.

    Args:
        S: contSet object to copy.

    Returns:
        A copied contSet object.

    Raises:
        CORAerror: Always raised, as this method must be overridden.
    """
    # This base function should not be called directly.
    # A subclass that is copyable must have its own `copy` method,
    # which will be found by Python's method resolution order before
    # the ContSet's recursive `copy` method is reached. If we get here,
    # it means no specific implementation was found.
    raise CORAerror('CORA:notSupported',
                   f'The class {S.__class__.__name__} does not support the copy operation.') 