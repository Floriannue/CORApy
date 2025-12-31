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
    # MATLAB evaluates this as a single expression
    # Compute step by step to ensure correct shapes
    f_mi = f(mi)
    # Extract numeric value if f_mi is interval
    if isinstance(f_mi, Interval):
        f_mi = f_mi.center()
    f_mi = np.asarray(f_mi).flatten()
    
    # A * mi
    A_mi = np.asarray(A @ mi).flatten()
    
    # f(mi) - A * mi (both are 1D numeric vectors)
    f_mi_minus_A_mi = f_mi - A_mi
    
    # (J - A) * (dom - mi)
    J_minus_A = J - A
    dom_minus_mi = dom - mi
    J_minus_A_dom = J_minus_A @ dom_minus_mi
    
    # Ensure J_minus_A_dom is 1D interval vector (flatten if 2D)
    # MATLAB: (J-A)*(dom-mi) should give (m,) interval vector where m = number of constraints
    if isinstance(J_minus_A_dom, Interval):
        if J_minus_A_dom.dim() > 1 or (J_minus_A_dom.inf.ndim > 1 or J_minus_A_dom.sup.ndim > 1):
            # Flatten to 1D
            num_constraints = A.shape[0]
            inf_flat = J_minus_A_dom.inf.flatten()
            sup_flat = J_minus_A_dom.sup.flatten()
            # Take first num_constraints elements
            if inf_flat.size >= num_constraints:
                inf_flat = inf_flat[:num_constraints]
                sup_flat = sup_flat[:num_constraints]
            J_minus_A_dom = Interval(inf_flat, sup_flat)
    
    # b = f(mi) - A * mi + (J - A) * (dom - mi)
    # numeric + interval = interval (handled by plus function)
    b = f_mi_minus_A_mi + J_minus_A_dom
    
    # Ensure b is 1D interval vector (MATLAB: b is (m,) interval vector)
    # b should have shape (m,) where m = number of constraints (rows of A)
    if isinstance(b, Interval):
        if b.dim() > 1 or (b.inf.ndim > 1 or b.sup.ndim > 1):
            num_constraints = A.shape[0]
            inf_flat = b.inf.flatten()
            sup_flat = b.sup.flatten()
            if inf_flat.size >= num_constraints:
                inf_flat = inf_flat[:num_constraints]
                sup_flat = sup_flat[:num_constraints]
            b = Interval(inf_flat, sup_flat)
    
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
                # MATLAB: b(j) extracts j-th element of interval vector b as scalar interval
                # MATLAB uses 'dom' which gets updated in place, so we use 'res' here
                # Extract j-th element of b (MATLAB: b(j))
                # b should be 1D interval vector, so b.inf[j] and b.sup[j] are scalars
                b_inf_j = b.inf[j] if b.inf.ndim == 1 else b.inf.flatten()[j]
                b_sup_j = b.sup[j] if b.sup.ndim == 1 else b.sup.flatten()[j]
                b_j = Interval(b_inf_j, b_sup_j)
                
                # MATLAB: a*dom where a is (1,n) row vector, dom is (n,1) interval
                # MATLAB: a*dom computes dot product: sum(a[i]*dom[i]) = scalar interval
                # Reshape a to (1, n) row vector for matrix multiplication
                a_row = a.reshape(1, -1) if a.ndim == 1 else a
                # Compute: (1,n) @ (n,1) interval = (1,1) interval -> scalar
                a_dom = a_row @ res
                # Ensure scalar (1D interval with dim=1)
                # The result should be scalar, but mtimes may return (1,1) which needs flattening
                if a_dom.dim() > 1 or (a_dom.inf.ndim > 1 or a_dom.sup.ndim > 1):
                    # Flatten and take the sum (for matrix multiplication result)
                    # If result is (1,1), flatten gives [value], take [0]
                    # If result is (1,n) incorrectly, sum over second dimension
                    inf_flat = a_dom.inf.flatten()
                    sup_flat = a_dom.sup.flatten()
                    # For dot product result, should be single value
                    # If multiple values, sum them (shouldn't happen for correct matrix mult)
                    if inf_flat.size > 1:
                        # Sum all elements (dot product result)
                        a_dom = Interval(np.sum(inf_flat), np.sum(sup_flat))
                    else:
                        a_dom = Interval(inf_flat[0] if inf_flat.size > 0 else 0.0,
                                         sup_flat[0] if sup_flat.size > 0 else 0.0)
                
                # MATLAB: temp = -(b(j) + a*dom)/A(j,i);
                temp = -(b_j + a_dom) / A[j, i]
                # Ensure scalar (1D interval with dim=1)
                if temp.dim() > 1 or (temp.inf.ndim > 1 or temp.sup.ndim > 1):
                    temp = Interval(temp.inf.flatten()[0] if temp.inf.size > 0 else 0.0,
                                    temp.sup.flatten()[0] if temp.sup.size > 0 else 0.0)
                
                # MATLAB: dom_ = dom(i) & temp;
                # MATLAB: dom(i) extracts i-th element of interval vector dom
                # MATLAB uses 'dom' which gets updated in place, so we use 'res' here
                dom_i = Interval(res.inf[i], res.sup[i])
                dom_ = dom_i & temp
                
                # MATLAB: if ~representsa_(dom_,'emptySet',eps)
                if not dom_.representsa_('emptySet', np.finfo(float).eps):
                    # MATLAB: dom(i) = dom_;
                    # Update res with the contracted interval
                    new_inf = res.inf.copy()
                    new_sup = res.sup.copy()
                    new_inf[i] = dom_.inf[0] if dom_.inf.size > 0 else dom_.inf
                    new_sup[i] = dom_.sup[0] if dom_.sup.size > 0 else dom_.sup
                    res = Interval(new_inf, new_sup)
                else:
                    # MATLAB: res = []; return;
                    return None
    
    return res

