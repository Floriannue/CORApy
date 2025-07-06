import numpy as np
from .interval import Interval

def subsasgn(I, S, val):
    """
    subsasgn - Overloads the operator that writes elements, e.g., I(1,2)=val,
    where the element of the first row and second column is referred to.

    Syntax:
        I = subsasgn(I,S,val)

    Inputs:
        I - interval object
        S - contains information of the type and content of element selections
        val - value to be inserted

    Outputs:
        I - interval object
    """

    # check if value is an interval
    if not isinstance(val, Interval):
        val = Interval(val, val)

    # check if parentheses are used to select elements
    I.inf[S] = val.inf
    I.sup[S] = val.sup
    
    return I 