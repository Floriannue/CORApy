import numpy as np
from .interval import Interval

def subsref(I, S):
    """
    subsref - Overloads the operator that selects elements, e.g., I(1,2),
    where the element of the first row and second column is referred to.

    Syntax:
        newObj = subsref(I,S)

    Inputs:
        I - interval object
        S - contains information of the type and content of element selections

    Outputs:
        newObj - element or elemets of the interval matrix
    """
    
    # obtain sub-intervals from the interval object
    new_inf = I.inf[S]
    new_sup = I.sup[S]
    
    return Interval(new_inf, new_sup) 