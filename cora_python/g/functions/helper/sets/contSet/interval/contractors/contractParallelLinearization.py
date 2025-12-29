"""
contractParallelLinearization - implementation of the parallel
                                linearization contractor acc. to 
                                Sec. 4.3.4 in [1]

Syntax:
    res = contractParallelLinearization(f,dom,jacHan)
    res = contractParallelLinearization(f,dom,jacHan,method)

Inputs:
    f - function handle for the constraint f(x) = 0
    dom - initial domain (class: interval)
    jacHan - function handle for the constraint jacobian matrix
    method - range bounding method ('interval' or 'taylm')

Outputs:
    res - contracted domain (class: interval)

Example: 
    f = @(x) [x(1)^2 - 4*x(2); 
              x(2)^2 - 2*x(1) + 4*x(2)];
    dom = interval([-0.1;-0.1],[0.1;0.1]);
   
    res = contract(f,dom,'linearize');

References:
    [1] L. Jaulin et al. "Applied Interval Analysis", 2006

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contract

Authors:       Zhuoling Li, Niklas Kochdumper
Written:       04-November-2019 
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, Callable, Any

from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from cora_python.contSet.polytope.polytope import Polytope


def contractParallelLinearization(f: Callable, dom: Interval, jacHan: Callable,
                                  method: Optional[str] = None) -> Optional[Interval]:
    """
    Implementation of the parallel linearization contractor
    
    Args:
        f: function handle for the constraint f(x) = 0
        dom: initial domain (interval object)
        jacHan: function handle for the constraint jacobian matrix
        method: range bounding method ('interval' or 'taylm'), default 'taylm'
        
    Returns:
        res: contracted domain (interval object) or None if empty
    """
    
    # Parse input arguments
    # MATLAB: method = 'taylm';
    # MATLAB: if nargin >= 4 && ~isempty(varargin{1})
    if method is None:
        method = 'taylm'
    
    # Apply mean value theorem to enclose constraints by parallel planes
    # MATLAB: mi = center(dom);
    mi = dom.center()
    # MATLAB: A = jacHan(mi);
    A = jacHan(mi)
    # Ensure A is 2D array
    A = np.asarray(A)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    elif A.ndim == 0:
        A = A.reshape(1, 1)
    
    # MATLAB: if strcmp(method,'taylm')
    if method == 'taylm':
        # MATLAB: try
        try:
            # MATLAB: tay = taylm(dom);
            from cora_python.contSet.taylm.taylm import Taylm
            tay = Taylm(dom)
            # MATLAB: J = interval(jacHan(tay));
            J_interval = Interval(jacHan(tay))
            J = J_interval
        except Exception:
            # MATLAB: catch
            # MATLAB: J = jacHan(dom);
            J = jacHan(dom)
            if not isinstance(J, Interval):
                J = Interval(J)
    else:
        # MATLAB: else
        # MATLAB: J = jacHan(dom);
        J = jacHan(dom)
        if not isinstance(J, Interval):
            J = Interval(J)
    
    # MATLAB: b = f(mi) - A * mi + (J - A) * (dom - mi);
    f_mi = f(mi)
    if isinstance(f_mi, Interval):
        f_mi_val = f_mi.center()
    else:
        f_mi_val = np.asarray(f_mi)
        # Ensure f_mi_val is a column vector (MATLAB returns column vectors)
        if f_mi_val.ndim == 1:
            f_mi_val = f_mi_val.reshape(-1, 1)
        elif f_mi_val.ndim == 0:
            f_mi_val = f_mi_val.reshape(1, 1)
    
    # MATLAB: A * mi
    A_mi = A @ mi if hasattr(A, '__matmul__') else np.dot(A, mi)
    A_mi = np.asarray(A_mi)
    # Ensure A_mi is a column vector
    if A_mi.ndim == 1:
        A_mi = A_mi.reshape(-1, 1)
    elif A_mi.ndim == 0:
        A_mi = A_mi.reshape(1, 1)
    
    # MATLAB: dom - mi
    dom_minus_mi = dom - mi
    
    # MATLAB: J - A
    if isinstance(J, Interval):
        J_center = J.center() if hasattr(J, 'center') else J
        J_minus_A = J_center - A
    else:
        J_minus_A = J - A
    
    # MATLAB: (J - A) * (dom - mi)
    # Use interval matrix multiplication via @ operator
    J_minus_A_dom = J_minus_A @ dom_minus_mi
    
    # MATLAB: b = f(mi) - A * mi + (J - A) * (dom - mi);
    # All should be column vectors or intervals
    if isinstance(J_minus_A_dom, Interval):
        # Extract inf and sup from interval and ensure they're column vectors
        # J_minus_A_dom should have shape matching the number of constraints (rows of A)
        J_minus_A_dom_inf = np.asarray(J_minus_A_dom.inf)
        J_minus_A_dom_sup = np.asarray(J_minus_A_dom.sup)
        # Flatten to 1D first, then reshape to column vector
        J_minus_A_dom_inf = J_minus_A_dom_inf.flatten()
        J_minus_A_dom_sup = J_minus_A_dom_sup.flatten()
        # Reshape to column vector with correct number of constraints
        num_constraints = A.shape[0]
        if J_minus_A_dom_inf.size != num_constraints:
            # If size doesn't match, take first num_constraints elements
            J_minus_A_dom_inf = J_minus_A_dom_inf[:num_constraints]
            J_minus_A_dom_sup = J_minus_A_dom_sup[:num_constraints]
        J_minus_A_dom_inf = J_minus_A_dom_inf.reshape(-1, 1)
        J_minus_A_dom_sup = J_minus_A_dom_sup.reshape(-1, 1)
        # Create new interval with column vectors
        b = Interval(J_minus_A_dom_inf.flatten(), J_minus_A_dom_sup.flatten())
        # Now add f_mi_val - A_mi (both should be column vectors)
        b = f_mi_val - A_mi + b
    else:
        J_minus_A_dom = np.asarray(J_minus_A_dom)
        # Flatten to 1D first
        J_minus_A_dom = J_minus_A_dom.flatten()
        # Ensure we have the correct number of constraints
        if J_minus_A_dom.size != num_constraints:
            # If size doesn't match, take first num_constraints elements or pad with zeros
            if J_minus_A_dom.size > num_constraints:
                J_minus_A_dom = J_minus_A_dom[:num_constraints]
            else:
                # Pad with zeros if needed (shouldn't happen in normal cases)
                pad_size = num_constraints - J_minus_A_dom.size
                J_minus_A_dom = np.concatenate([J_minus_A_dom, np.zeros(pad_size)])
        # Reshape to column vector
        J_minus_A_dom = J_minus_A_dom.reshape(-1, 1)
        b = f_mi_val - A_mi + J_minus_A_dom
        # Ensure b is a column vector with correct number of constraints
        b = b.flatten()
        if b.size != num_constraints:
            if b.size > num_constraints:
                b = b[:num_constraints]
            else:
                pad_size = num_constraints - b.size
                b = np.concatenate([b, np.zeros(pad_size)])
        # Create interval from column vector
        b = Interval(b, b)
    
    # Solve linear program to compute new bounds for each variable
    # MATLAB: A_ = [-A;A];
    # Ensure A is 2D
    if A.ndim == 1:
        A = A.reshape(1, -1)
    A_ = np.vstack([-A, A])
    
    # MATLAB: b_ = [supremum(b);-infimum(b)];
    # In MATLAB, supremum(b) and infimum(b) return column vectors
    # Get b.sup and b.inf and ensure they're 1D arrays first
    b_sup = b.sup.flatten() if b.sup.ndim > 1 else b.sup
    b_inf = b.inf.flatten() if b.inf.ndim > 1 else b.inf
    # Reshape to column vectors
    b_sup = b_sup.reshape(-1, 1)
    b_inf = b_inf.reshape(-1, 1)
    # MATLAB: b_ = [supremum(b);-infimum(b)];
    b_ = np.vstack([b_sup, -b_inf])
    
    # Verify A_ and b_ have matching constraint counts
    # A_ should have 2 * num_constraints rows, b_ should have 2 * num_constraints rows
    if A_.shape[0] != b_.shape[0]:
        raise ValueError(f"Constraint count mismatch: A_ has {A_.shape[0]} rows (from A with {A.shape[0]} constraints), b_ has {b_.shape[0]} rows (from b with {b_sup.shape[0]} constraints). A shape: {A.shape}, b shape: {b.sup.shape if hasattr(b, 'sup') else 'N/A'}")
    
    # MATLAB: infi = infimum(dom);
    # MATLAB: sup = supremum(dom);
    # In MATLAB, infimum(dom) and supremum(dom) return column vectors
    infi = dom.inf
    sup = dom.sup
    # Ensure column vectors (MATLAB returns column vectors)
    if infi.ndim == 1:
        infi = infi.reshape(-1, 1)
    if sup.ndim == 1:
        sup = sup.reshape(-1, 1)
    
    # MATLAB: problem.Aineq = A_;
    # MATLAB: problem.bineq = b_;
    # MATLAB: problem.Aeq = [];
    # MATLAB: problem.beq = [];
    # MATLAB: problem.options = optimoptions('linprog','Display','off', ...
    #                           'ConstraintTolerance',1e-9);
    # Note: Python CORAlinprog doesn't use options, so we skip it
    
    # MATLAB: n = length(mi);
    n = len(mi)
    # MATLAB: f = zeros(n,1);
    f_obj = np.zeros((n, 1))
    
    # MATLAB: for i = 1:n
    for i in range(n):
        # MATLAB: f_ = f;
        # MATLAB: f_(i) = 1;
        f_ = f_obj.copy()
        f_[i] = 1
        
        # Compute new infimum
        # MATLAB: problem.f = f_;
        # MATLAB: problem.lb = infi;
        # MATLAB: problem.ub = sup;
        problem = {
            'f': f_.flatten(),  # CORAlinprog expects 1D array
            'Aineq': A_,
            'bineq': b_.flatten(),  # CORAlinprog expects 1D array
            'Aeq': np.array([]).reshape(0, n),
            'beq': np.array([]).reshape(0, 1),
            'lb': infi.flatten(),  # CORAlinprog expects 1D array
            'ub': sup.flatten()  # CORAlinprog expects 1D array
        }
        
        # MATLAB: [~,temp] = CORAlinprog(problem);
        x, fval, exitflag, output, lambda_out = CORAlinprog(problem)
        
        # MATLAB: if ~isempty(temp)
        if x is not None and fval is not None:
            # MATLAB: infi(i) = max(infi(i),temp);
            infi[i] = max(infi[i], fval)
        else:
            # MATLAB: else
            # MATLAB: poly = polytope([A_;eye(n);-eye(n)],[b_;sup;-infi]);
            # In MATLAB, all vectors are column vectors, so stacking works directly
            A_poly = np.vstack([A_, np.eye(n), -np.eye(n)])
            # Ensure all are column vectors before stacking
            b_col = b_ if b_.ndim == 2 and b_.shape[1] == 1 else b_.reshape(-1, 1)
            sup_col = sup if sup.ndim == 2 and sup.shape[1] == 1 else sup.reshape(-1, 1)
            infi_col = infi if infi.ndim == 2 and infi.shape[1] == 1 else infi.reshape(-1, 1)
            b_poly = np.vstack([b_col, sup_col, -infi_col])
            # Verify shapes match before creating polytope
            # MATLAB: poly = polytope([A_;eye(n);-eye(n)],[b_;sup;-infi]);
            # A_poly should have: A_.shape[0] + n + n rows
            # b_poly should have: b_.shape[0] + n + n rows
            expected_A_rows = A_.shape[0] + n + n
            expected_b_rows = b_.shape[0] + n + n
            if A_poly.shape[0] != b_poly.shape[0]:
                raise ValueError(f"Shape mismatch: A_poly has {A_poly.shape[0]} rows (expected {expected_A_rows}), b_poly has {b_poly.shape[0]} rows (expected {expected_b_rows}). A_ shape: {A_.shape}, b_ shape: {b_.shape}, sup shape: {sup.shape}, infi shape: {infi.shape}, n: {n}")
            poly = Polytope(A_poly, b_poly)
            
            # MATLAB: if representsa_(poly,'emptySet',eps)
            if poly.representsa_('emptySet', np.finfo(float).eps):
                # MATLAB: res = [];
                # MATLAB: return;
                return None
        
        # Compute new supremum
        # MATLAB: problem.f = -f_;
        # MATLAB: problem.lb = infi;
        # MATLAB: problem.ub = sup;
        problem['f'] = -f_.flatten()
        problem['lb'] = infi.flatten()
        problem['ub'] = sup.flatten()
        
        # MATLAB: [~,temp] = CORAlinprog(problem);
        x, fval, exitflag, output, lambda_out = CORAlinprog(problem)
        
        # MATLAB: if ~isempty(temp)
        if x is not None and fval is not None:
            # MATLAB: sup(i) = min(sup(i),-temp);
            sup[i] = min(sup[i], -fval)
        else:
            # MATLAB: else
            # MATLAB: poly = polytope([A_;eye(n);-eye(n)],[b_;sup;-infi]);
            A_poly = np.vstack([A_, np.eye(n), -np.eye(n)])
            # Ensure all are column vectors before stacking
            b_col = b_ if b_.ndim == 2 and b_.shape[1] == 1 else b_.reshape(-1, 1)
            sup_col = sup if sup.ndim == 2 and sup.shape[1] == 1 else sup.reshape(-1, 1)
            infi_col = infi if infi.ndim == 2 and infi.shape[1] == 1 else infi.reshape(-1, 1)
            b_poly = np.vstack([b_col, sup_col, -infi_col])
            poly = Polytope(A_poly, b_poly)
            
            # MATLAB: if representsa_(poly,'emptySet',eps)
            if poly.representsa_('emptySet', np.finfo(float).eps):
                # MATLAB: res = [];
                # MATLAB: return;
                return None
    
    # MATLAB: res = interval(infi,sup);
    # Ensure infi and sup are 1D arrays for Interval constructor
    infi_flat = infi.flatten() if infi.ndim > 1 else infi
    sup_flat = sup.flatten() if sup.ndim > 1 else sup
    res = Interval(infi_flat, sup_flat)
    return res
