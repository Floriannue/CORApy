from .interval import Interval
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interval import Interval

def enclose(I1: 'Interval', I2: 'Interval'):
    """
    enclose - encloses an interval and its affine transformation (calling
              convHull_ as the operation is equivalent for intervals)
    
    Syntax:
       I_out = enclose(I1, I2)
    
    Inputs:
       I1 - interval object
       I2 - interval object
    
    Outputs:
       I_out - interval object
    """
    
    # compute result via convHull
    return I1.convHull_(I2) 