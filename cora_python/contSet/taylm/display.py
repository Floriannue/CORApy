"""
display - Displays the properties of a taylm object on the command window

Syntax:
    display(obj)

Inputs:
    tay - taylm object

Outputs:
    ---

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Dmitry Grebenyuk, Niklas Kochdumper
Written:       31-March-2016
Last update:   18-July-2017 (DG, multivariable polynomial pack is added)
               11-April-2018 (NK, sort variables in the polynomial part)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.contSet.taylm.taylm import Taylm
from cora_python.contSet.interval.interval import Interval

if TYPE_CHECKING:
    from .taylm import Taylm


def _aux_displayPoly(obj: 'Taylm') -> str:
    """
    Auxiliary function to display polynomial part of taylm
    
    Args:
        obj: Taylm object
        
    Returns:
        str: Formatted polynomial string
    """
    try:
        import sympy as sp
    except ImportError:
        # Fallback if sympy not available
        return str(obj.coefficients)
    
    # Get coefficients
    c = obj.coefficients
    
    # Get monomials
    if hasattr(obj, 'monomials') and obj.monomials is not None:
        degs = obj.monomials[:, 1:] if obj.monomials.shape[1] > 1 else np.array([]).reshape(len(c), 0)
    else:
        degs = np.array([]).reshape(len(c), 0)
    
    # Get var names
    names = obj.names_of_var if hasattr(obj, 'names_of_var') else []
    
    if len(names) == 0 or degs.shape[1] == 0:
        # Simple case: no variables or no monomials
        return str(c[0]) if len(c) > 0 else "0"
    
    # Sort var names alphabetically
    names_array = np.array(names)
    ind = np.argsort(names_array)
    namesSort = names_array[ind]
    
    # Adapt exponent matrix to the sorted variable names
    degs = degs[:, ind]
    
    # Sort the exponent matrix according to the polynomial order of the terms
    # Sort by sum of degrees (descending), then by reversed degrees
    degs_sum = np.sum(degs, axis=1)
    sort_key = np.column_stack([-degs_sum, -degs[:, ::-1]])  # Negative for descending
    sort_ind = np.lexsort(sort_key.T[::-1])
    degs = degs[sort_ind, :]
    c = c[sort_ind]
    
    # Transform the var names to sympy symbols
    v = [sp.Symbol(name) for name in namesSort]
    
    # Make a sympy expression
    terms = []
    for i in range(len(c)):
        if c[i] == 0:
            continue
        term = c[i]
        for j, exp in enumerate(degs[i, :]):
            if exp > 0:
                term = term * (v[j] ** int(exp))
        terms.append(term)
    
    if len(terms) == 0:
        return "0"
    
    # Combine terms
    expr = sum(terms)
    
    # Format with limited precision (like MATLAB's vpa)
    try:
        expr_str = str(sp.N(expr, 5))
    except:
        expr_str = str(expr)
    
    return expr_str


def display_(tay: 'Taylm', var_name: str = None) -> str:
    """
    Displays the properties of a taylm object (internal function that returns string)
    
    Args:
        tay: taylm object
        var_name: Optional variable name
        
    Returns:
        str: String representation
    """
    lines = []
    
    # Display input variable
    lines.append("")
    if var_name is None:
        var_name = 'tay'
    lines.append(f"{var_name} =")
    lines.append("")
    
    # Display dimension (not inheriting from contSet)
    class_name = tay.__class__.__name__
    lines.append(f"{class_name}:")
    
    dim_val = tay.dim() if hasattr(tay, 'dim') and callable(tay.dim) else 0
    lines.append(f"- dimension: {dim_val}")
    lines.append("")
    
    # Determine the shape of a matrix
    if hasattr(tay, 'shape'):
        mi, mj = tay.shape
    else:
        # Try to infer shape from coefficients or monomials
        if hasattr(tay, 'coefficients') and tay.coefficients is not None:
            mi, mj = 1, 1  # Default to scalar
        else:
            mi, mj = 1, 1
    
    # Display each element
    for i in range(mi):
        rowStr_parts = []
        for j in range(mj):
            # Get the element (for now, assume single element or handle indexing)
            if mi == 1 and mj == 1:
                elem = tay
            else:
                # For matrix taylm, would need indexing - simplified for now
                elem = tay
            
            # Get a polynomial part; show rational numbers as decimals with 5 digits
            poly = _aux_displayPoly(elem)
            
            # Get an interval part
            if hasattr(elem, 'remainder') and elem.remainder is not None:
                remainder = elem.remainder
                if isinstance(remainder, Interval):
                    inf_val = remainder.inf if hasattr(remainder, 'inf') else 0
                    sup_val = remainder.sup if hasattr(remainder, 'sup') else 0
                    if isinstance(inf_val, np.ndarray) and inf_val.size > 0:
                        inf_val = float(inf_val.flatten()[0])
                    if isinstance(sup_val, np.ndarray) and sup_val.size > 0:
                        sup_val = float(sup_val.flatten()[0])
                    remainder_str = f"[{inf_val:.4f},{sup_val:.4f}]"
                else:
                    remainder_str = str(remainder)
            else:
                remainder_str = "[0.0000,0.0000]"
            
            # Add both parts
            str_elem = f"{poly} + {remainder_str}"
            
            # Add to a row
            rowStr_parts.append(str_elem)
        
        lines.append("\t".join(rowStr_parts))
    
    lines.append("")
    
    return "\n".join(lines)


def display(tay: 'Taylm', var_name: str = None) -> None:
    """
    Displays the properties of a taylm object (prints to stdout)
    
    Args:
        tay: taylm object
        var_name: Optional variable name
    """
    print(display_(tay, var_name), end='')

