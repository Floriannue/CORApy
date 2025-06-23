"""
display - Displays the properties of an interval object (lower and upper
   bounds) on the command window

Syntax:
   display(I)

Inputs:
   I - interval object

Outputs:
   ---

Example: 
   I = Interval(2, 3)
   display(I)

Authors:       Matthias Althoff
Written:       19-June-2015
Last update:   22-February-2016 (DG, now it displays the name)
               01-May-2020 (MW, handling of empty case)
               11-September-2023 (TL, respect output display format)
               27-September-2024 (MW, all-zero sparse case)
               18-October-2024 (TL, n-d intervals)
Last revision: ---
"""

import numpy as np
from typing import TYPE_CHECKING
from scipy.sparse import issparse

from .interval import Interval


def display(I: Interval, name: str = None) -> str:
    """
    Displays the properties of an interval object, mirroring MATLAB behavior.
    
    Args:
        I: Interval object
        name: Variable name to display
        
    Returns:
        String representation for display
    """
    output_lines = []
    
    # If no name provided, use default
    if name is None:
        name = "ans"
    
    # Special cases (only vector) - check if it's 1D or 2D with one dimension <= 1
    dims = I.inf.shape
    is_vector = len(dims) <= 2 and (dims[1] <= 1 if len(dims) > 1 else True)
    
    if is_vector:
        if I.representsa_('emptySet'):
            return _disp_empty_set(I, name)
        elif I.representsa_('fullspace'):
            return _disp_rn(I, name)
    
    # All-zero and sparse
    if issparse(I.inf) and I.representsa_('origin', 0):
        output_lines.append("")
        output_lines.append(f"{name} =")
        output_lines.append("")
        output_lines.append(f"   All zero sparse interval: {'x'.join(map(str, I.inf.shape))}")
        output_lines.append("")
        return "\n".join(output_lines)
    
    # Standard display
    output_lines.append("")
    output_lines.append(f"{name} =")
    output_lines.append("")
    
    # Display dimension
    output_lines.append(f"Interval object with dimension: {I.dim()}")
    output_lines.append("")
    
    # Check dimension for display method
    if len(dims) <= 2:
        # Display 2-dimensional interval
        output_lines.extend(_aux_display_2d(I))
    else:
        # Display n-dimensional interval page-wise
        output_lines.extend(_aux_display_nd(I, name))
    
    return "\n".join(output_lines)


def _disp_empty_set(I: Interval, name: str) -> str:
    """Display text for empty interval objects"""
    output_lines = []
    output_lines.append("")
    output_lines.append(f"{name} =")
    output_lines.append("")
    output_lines.append(f"  {I.dim()}-dimensional empty set (represented as Interval)")
    output_lines.append("")
    return "\n".join(output_lines)


def _disp_rn(I: Interval, name: str) -> str:
    """Display text for fullspace interval objects"""
    output_lines = []
    output_lines.append("")
    output_lines.append(f"{name} =")
    output_lines.append("")
    output_lines.append(f"  R^{I.dim()} (represented as Interval)")
    output_lines.append("")
    return "\n".join(output_lines)


def _aux_display_2d(I: Interval) -> list:
    """Display 2-dimensional interval"""
    return _display_interval_core(I)


def _aux_display_nd(I: Interval, varname: str) -> list:
    """Display n-dimensional interval page-wise"""
    output_lines = []
    
    # Determine number of pages
    dims = I.inf.shape
    dims_pages = dims[2:]
    num_pages = np.prod(dims_pages)
    
    # Reshape for page-wise access
    I_pages_inf = I.inf.reshape(dims[0], dims[1], num_pages)
    I_pages_sup = I.sup.reshape(dims[0], dims[1], num_pages)
    
    # Function to convert linear indices to matrix indices
    def convert_index(i):
        indices = []
        remaining = i - 1
        for d in range(len(dims_pages)):
            divisor = np.prod(dims_pages[:d]) if d > 0 else 1
            indices.append((remaining // divisor) % dims_pages[d] + 1)
        return indices
    
    for i in range(num_pages):
        # Convert indices
        page_idx = convert_index(i + 1)
        
        # Display input variable
        page_idx_str = ','.join(map(str, page_idx))
        output_lines.append(f"{varname}(:,:,{page_idx_str}) = ")
        output_lines.append("")
        
        # Create interval for current page
        page_interval = Interval(I_pages_inf[:, :, i], I_pages_sup[:, :, i])
        
        # Display current page
        output_lines.extend(_display_interval_core(page_interval))
    
    return output_lines


def _display_interval_core(I: Interval) -> list:
    """
    Core function to display interval bounds in matrix format.
    This is the equivalent of displayInterval helper function.
    """
    output_lines = []
    
    if issparse(I.inf):
        # Handle sparse intervals
        
        # Non-zero indices of infimum
        inf_rows, inf_cols = I.inf.nonzero()
        # Non-zero indices of supremum
        sup_rows, sup_cols = I.sup.nonzero()
        
        # For sparse matrices, we plot all indices where either the infimum
        # or the supremum is non-zero
        inf_indices = set(zip(inf_rows, inf_cols))
        sup_indices = set(zip(sup_rows, sup_cols))
        indices = sorted(inf_indices.union(sup_indices))
        
        # Loop over all indices
        for row, col in indices:
            # Set index
            idx_str = f"({row+1},{col+1})"  # MATLAB uses 1-based indexing
            # Values
            lb = float(I.inf[row, col])
            ub = float(I.sup[row, col])
            # Format values
            lb_str = _format_number(lb)
            ub_str = _format_number(ub)
            output_lines.append(f"  {idx_str:<8} [{lb_str}, {ub_str}]")
    
    else:
        # Handle dense intervals
        
        # Determine size of interval
        if I.inf.ndim == 1:
            rows, cols = I.inf.shape[0], 1
        else:
            rows, cols = I.inf.shape
        
        for i in range(rows):
            row_strs = []
            # Display one row
            for j in range(cols):
                if I.inf.ndim == 1:
                    lb = I.inf[i] if cols == 1 else I.inf[i, j]
                    ub = I.sup[i] if cols == 1 else I.sup[i, j]
                else:
                    lb = I.inf[i, j]
                    ub = I.sup[i, j]
                lb_str = _format_number(lb)
                ub_str = _format_number(ub)
                interval_str = f"[{lb_str}, {ub_str}]"
                row_strs.append(interval_str)
            
            # Join row elements with spaces
            row_line = "  " + "  ".join(row_strs)
            output_lines.append(row_line)
    
    output_lines.append("")
    return output_lines


def _format_number(num: float) -> str:
    """
    Format a number for display, similar to MATLAB's formattedDisplayText.
    """
    if np.isnan(num):
        return "NaN"
    elif np.isinf(num):
        return "Inf" if num > 0 else "-Inf"
    elif num == 0:
        return "0"
    elif abs(num) >= 1e4 or (abs(num) < 1e-3 and num != 0):
        # Use scientific notation for very large or very small numbers
        return f"{num:.4e}"
    else:
        # Use fixed-point notation, remove trailing zeros
        formatted = f"{num:.4f}".rstrip('0').rstrip('.')
        return formatted if formatted else "0" 