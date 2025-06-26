from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid

def copy(E: 'Ellipsoid') -> 'Ellipsoid':
    """
    copy - copies the ellipsoid object

    Syntax:
        E_out = copy(E)

    Inputs:
        E - ellipsoid object

    Outputs:
        E_out - copied ellipsoid object
    """
    # call copy constructor
    from .ellipsoid import Ellipsoid as EllipsoidClass
    return EllipsoidClass(E) 