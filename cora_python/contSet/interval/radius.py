import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interval import Interval


def radius(I: 'Interval') -> float:
    """
    radius - Computes radius of enclosing hyperball of an interval 

    Syntax:
        r = radius(I)

    Inputs:
        I - interval object

    Outputs:
        r - radius of enclosing hyperball

    Example:
        I = Interval(np.array([[-2], [1]]), np.array([[4], [3]]))
        r = radius(I)
    """
    # rad(I) returns (sup - inf) / 2 which is a vector of radii
    r_vec = I.rad()
    # r = sqrt(sum(rad(I).^2,"all"));
    r = np.sqrt(np.sum(r_vec**2))
    return float(r) 