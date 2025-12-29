"""
contractInterval - contraction based on interval arithmetic

Syntax:
    res = contractInterval(f,dom,jacHan)
    res = contractInterval(f,dom,jacHan,method)

Inputs:
    f - function handle for the constraint f(x) = 0
    dom - initial domain (class: interval)
    jacHan - function handle for the constraint jacobian matrix
    method - range bounding method ('interval' or 'taylm')

Outputs:
    res - contracted domain (class: interval)

Example: 
    f = @(x) [x(1)^2 - 4*x(2); 
              x(2)^2 - 2*x(1) + 4*x(2)]);
    dom = interval([-0.1;-0.1],[0.1;0.1]);
   
    res = contract(f,dom,'interval');

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contract

Authors:       Niklas Kochdumper
Written:       17-December-2020 
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, Callable, Any

from cora_python.contSet.interval.interval import Interval


def contractInterval(f: Callable, dom: Interval, jacHan: Callable, 
                     method: Optional[str] = None) -> Interval:
    """
    Contraction based on interval arithmetic
    
    Args:
        f: function handle for the constraint f(x) = 0
        dom: initial domain (interval object)
        jacHan: function handle for the constraint jacobian matrix
        method: range bounding method ('interval' or 'taylm'), default 'taylm'
        
    Returns:
        res: contracted domain (interval object)
    """
    
    # Parse input arguments
    if method is None:
        method = 'taylm'
    
    # Apply mean value theorem to enclose constraints by parallel planes
    mi = dom.center()
    A = jacHan(mi)
    
    if method == 'taylm':
        try:
            from cora_python.contSet.taylm.taylm import Taylm
            tay = Taylm(dom)
            J_interval = Interval(jacHan(tay))
            J = J_interval
        except Exception:
            J = jacHan(dom)  # jacHan should return an interval when given an interval
            if not isinstance(J, Interval):
                # If jacHan doesn't return an interval, wrap it
                J = Interval(J)
    else:
        J = jacHan(dom)
        if not isinstance(J, Interval):
            J = Interval(J)
    
    # MATLAB: b = f(mi) - A * mi + (J - A) * (dom - mi);
    # Note: f(mi) should return a vector, A*mi is matrix-vector product
    # (J - A) is interval matrix subtraction, (dom - mi) is interval subtraction
    f_mi = f(mi)
    if isinstance(f_mi, Interval):
        f_mi_val = f_mi.center()  # Use center for point evaluation
    else:
        f_mi_val = np.asarray(f_mi)
        # Ensure column vector (MATLAB returns column vectors)
        if f_mi_val.ndim == 1:
            f_mi_val = f_mi_val.reshape(-1, 1)
        elif f_mi_val.ndim == 0:
            f_mi_val = f_mi_val.reshape(1, 1)
    
    # MATLAB: A * mi
    A_mi = A @ mi if hasattr(A, '__matmul__') else np.dot(A, mi)
    A_mi = np.asarray(A_mi)
    # Ensure column vector
    if A_mi.ndim == 1:
        A_mi = A_mi.reshape(-1, 1)
    elif A_mi.ndim == 0:
        A_mi = A_mi.reshape(1, 1)
    
    # MATLAB: dom - mi
    dom_minus_mi = dom - mi
    
    # MATLAB: J - A
    # J is an interval matrix, A is a regular matrix
    if isinstance(J, Interval):
        # J is interval, need to handle interval-regular matrix subtraction
        # Use center of J for the subtraction
        J_center = J.center() if hasattr(J, 'center') else J
        J_minus_A = J_center - A
    else:
        J_minus_A = J - A
    
    # MATLAB: (J - A) * (dom - mi)
    # J_minus_A is (m x n), dom_minus_mi is (n x 1) interval
    # Use interval matrix multiplication via @ operator
    J_minus_A_dom = J_minus_A @ dom_minus_mi
    
    # Ensure J_minus_A_dom is 1D interval vector (flatten if 2D)
    if isinstance(J_minus_A_dom, Interval):
        # Flatten to 1D if it's 2D (result of matrix multiplication)
        if J_minus_A_dom.dim() > 1 or (J_minus_A_dom.inf.ndim > 1 or J_minus_A_dom.sup.ndim > 1):
            J_minus_A_dom = Interval(J_minus_A_dom.inf.flatten(), J_minus_A_dom.sup.flatten())
    
    # MATLAB: b = f(mi) - A * mi + (J - A) * (dom - mi);
    # Compute step by step to match MATLAB evaluation order
    # First: f(mi) - A * mi (both should be column vectors)
    f_mi_minus_A_mi = f_mi_val - A_mi
    # Ensure column vector
    if f_mi_minus_A_mi.ndim == 1:
        f_mi_minus_A_mi = f_mi_minus_A_mi.reshape(-1, 1)
    elif f_mi_minus_A_mi.ndim == 0:
        f_mi_minus_A_mi = f_mi_minus_A_mi.reshape(1, 1)
    
    # Then: (f(mi) - A * mi) + (J - A) * (dom - mi)
    # Convert numeric to interval and add
    if isinstance(J_minus_A_dom, Interval):
        # J_minus_A_dom is already an interval, add numeric vector to it
        # Convert f_mi_minus_A_mi to interval and add
        b = Interval(f_mi_minus_A_mi.flatten(), f_mi_minus_A_mi.flatten()) + J_minus_A_dom
    else:
        # J_minus_A_dom is numeric, convert to interval and add
        J_minus_A_dom_arr = np.asarray(J_minus_A_dom)
        if J_minus_A_dom_arr.ndim == 1:
            J_minus_A_dom_arr = J_minus_A_dom_arr.reshape(-1, 1)
        elif J_minus_A_dom_arr.ndim == 0:
            J_minus_A_dom_arr = J_minus_A_dom_arr.reshape(1, 1)
        # Ensure shapes match
        if f_mi_minus_A_mi.shape != J_minus_A_dom_arr.shape:
            # Reshape to match
            min_size = min(f_mi_minus_A_mi.size, J_minus_A_dom_arr.size)
            f_mi_minus_A_mi = f_mi_minus_A_mi.flatten()[:min_size].reshape(-1, 1)
            J_minus_A_dom_arr = J_minus_A_dom_arr.flatten()[:min_size].reshape(-1, 1)
        b = Interval(f_mi_minus_A_mi.flatten(), f_mi_minus_A_mi.flatten()) + Interval(J_minus_A_dom_arr.flatten(), J_minus_A_dom_arr.flatten())
    
    # Ensure b is 1D interval vector
    if b.dim() > 1 or (b.inf.ndim > 1 or b.sup.ndim > 1):
        b = Interval(b.inf.flatten(), b.sup.flatten())
    
    # Loop over all variables
    res = dom
    
    for i in range(dom.dim()):
        # Loop over all constraints
        for j in range(A.shape[0]):
            if abs(A[j, i]) > 1e-10:
                # Contract interval domain based on current constraints
                # MATLAB: a = A(j,:); a(i) = 0;
                a = A[j, :].copy()
                a[i] = 0
                
                # MATLAB: temp = -(b(j) + a*dom)/A(j,i);
                # b(j) is an interval, a*dom is interval-vector product
                # Note: MATLAB uses 'dom' which gets updated in place, so we use 'res' here
                # Extract j-th element of b as scalar interval
                if isinstance(b, Interval):
                    # b should be a 1D interval vector, extract j-th element
                    b_inf_flat = b.inf.flatten()
                    b_sup_flat = b.sup.flatten()
                    b_j = Interval(b_inf_flat[j] if j < len(b_inf_flat) else b_inf_flat[0],
                                   b_sup_flat[j] if j < len(b_sup_flat) else b_sup_flat[0])
                else:
                    b_val = b[j] if hasattr(b, '__getitem__') and len(b) > j else (b if np.isscalar(b) else b[0])
                    b_j = Interval(b_val, b_val)
                
                # a*res: a is (1, n) row vector, res is (n,) interval vector
                # Result should be scalar interval (1D with dim=1)
                a_dom = a @ res if hasattr(a, '__matmul__') else np.dot(a, res)
                if not isinstance(a_dom, Interval):
                    a_dom = Interval(a_dom, a_dom)
                # Ensure a_dom is scalar (1D interval with dim=1)
                # Flatten if 2D (result of matrix multiplication)
                if a_dom.dim() > 1 or (a_dom.inf.ndim > 1 or a_dom.sup.ndim > 1):
                    a_dom = Interval(a_dom.inf.flatten(), a_dom.sup.flatten())
                # If still not 1D, extract first element
                if a_dom.dim() != 1:
                    a_dom_inf = a_dom.inf.flatten()[0] if a_dom.inf.size > 0 else 0.0
                    a_dom_sup = a_dom.sup.flatten()[0] if a_dom.sup.size > 0 else 0.0
                    a_dom = Interval(a_dom_inf, a_dom_sup)
                
                temp = -(b_j + a_dom) / A[j, i]
                # Ensure temp is scalar (1D interval with dim=1)
                # Flatten if 2D (result of arithmetic operations)
                if temp.dim() > 1 or (temp.inf.ndim > 1 or temp.sup.ndim > 1):
                    temp = Interval(temp.inf.flatten(), temp.sup.flatten())
                # If still not 1D, extract first element
                if temp.dim() != 1:
                    temp_inf = temp.inf.flatten()[0] if temp.inf.size > 0 else 0.0
                    temp_sup = temp.sup.flatten()[0] if temp.sup.size > 0 else 0.0
                    temp = Interval(temp_inf, temp_sup)
                
                # MATLAB: dom_ = dom(i) & temp;
                # Note: MATLAB uses dom(i) which is the current (updated) value, so we use res(i)
                dom_i = Interval(res.inf[i], res.sup[i])
                dom_ = dom_i & temp  # Interval intersection
                
                # Debug: Check if intersection is valid
                # In MATLAB, if dom_ is empty, we return []
                if dom_.representsa_('emptySet', np.finfo(float).eps):
                    # Empty set - return None (MATLAB returns [])
                    return None
                
                # Update res with the contracted interval
                # Update dom(i) - MATLAB: dom(i) = dom_;
                # Create new interval with updated dimension
                new_inf = res.inf.copy()
                new_sup = res.sup.copy()
                # Extract scalar values from dom_
                dom_inf_val = dom_.inf if np.isscalar(dom_.inf) else (dom_.inf[0] if dom_.inf.size > 0 else dom_.inf.item() if hasattr(dom_.inf, 'item') else float(dom_.inf))
                dom_sup_val = dom_.sup if np.isscalar(dom_.sup) else (dom_.sup[0] if dom_.sup.size > 0 else dom_.sup.item() if hasattr(dom_.sup, 'item') else float(dom_.sup))
                new_inf[i] = dom_inf_val
                new_sup[i] = dom_sup_val
                res = Interval(new_inf, new_sup)
    
    return res

