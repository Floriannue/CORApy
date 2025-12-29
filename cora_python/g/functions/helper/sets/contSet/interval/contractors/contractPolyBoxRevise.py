"""
contractPolyBoxRevise - implementation of the contractor based on 
                        extremal functions acc. to [1]

Syntax:
    res = contractPolyBoxRevise(f,dom)

Inputs:
    f - function handle for the constraint f(x) = 0
    dom - initial domain (class: interval)

Outputs:
    res - contracted domain (class: interval)

Example: 
    f = @(x) (1 + 0.1*x(2))*(x(1) + 0.05*x(1)^3) + 0.2*x(2);
    dom = interval([-1;-1],[1;1]);
   
    res = contract(f,dom,'polynomial');

References:
    [1] G. Trombettoni et al. "A Box-Consistency Contractor Based on 
        Extremal Functions", 2010

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
import sympy as sp
import itertools
from typing import TYPE_CHECKING, Optional, Callable, Union, List, Any

from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .contractForwardBackward import contractForwardBackward


def contractPolyBoxRevise(f: Callable, dom: Interval) -> Interval:
    """
    Implementation of the contractor based on extremal functions
    
    Args:
        f: function handle for the constraint f(x) = 0
        dom: initial domain (interval object)
        
    Returns:
        res: contracted domain (interval object)
    """
    # Initialization
    n = dom.dim()
    
    # Create symbolic variables
    vars_sym = sp.symbols('x0:{}'.format(n))
    if n == 1:
        vars_sym = [vars_sym]
    else:
        vars_sym = list(vars_sym)
    
    # Evaluate function with symbolic variables
    p = f(vars_sym)
    
    # Handle if p is scalar or array
    if not isinstance(p, (list, tuple, np.ndarray)):
        p = [p]
    elif isinstance(p, np.ndarray):
        p = p.flatten().tolist()
    
    mins = dom.inf.copy()
    maxs = dom.sup.copy()
    
    # Check if function is a polynomial
    # MATLAB: try sym2poly(p(i)); catch ex; if strcmp(ex.identifier,'symbolic:sym:sym2poly:errmsg2')
    for i in range(len(p)):
        # Try to convert to polynomial coefficients
        # sympy's Poly will raise PolynomialError if not a polynomial
        # MATLAB: sym2poly(p(i)) raises error if not polynomial
        try:
            sp.Poly(p[i], vars_sym)
        except (sp.PolynomialError, TypeError):
            # MATLAB: throw(CORAerror('CORA:specialError',...))
            raise CORAerror('CORA:specialError',
                           'Contractor "polynomial" is only applicable for polynomial constraints!')
    
    # Execute contraction for each variable
    for i in range(n):
        # Loop over all constraints
        for j in range(len(p)):
            # Test if a split is needed
            dom_i_inf = dom.inf[i] if dom.inf.size > i else dom.inf
            dom_i_sup = dom.sup[i] if dom.sup.size > i else dom.sup
            
            if dom_i_inf < 0 and dom_i_sup > 0:
                # Contract first splitted domain
                tmp = dom
                int1 = Interval(dom_i_inf, 0)
                new_inf = dom.inf.copy()
                new_sup = dom.sup.copy()
                new_inf[i] = dom_i_inf
                new_sup[i] = 0
                dom_split1 = Interval(new_inf, new_sup)
                res1_ = _aux_contractBox(vars_sym[i], p[j], dom_split1, i, vars_sym)
                
                # Contract second splitted domain
                int2 = Interval(0, dom_i_sup)
                new_inf = dom.inf.copy()
                new_sup = dom.sup.copy()
                new_inf[i] = 0
                new_sup[i] = dom_i_sup
                dom_split2 = Interval(new_inf, new_sup)
                res2_ = _aux_contractBox(vars_sym[i], p[j], dom_split2, i, vars_sym)
                dom = tmp
                
                # Combine the results
                res_ = Interval(min(res1_), max(res2_))
                
                mins[i] = res_.inf if np.isscalar(res_.inf) else res_.inf[0]
                maxs[i] = res_.sup if np.isscalar(res_.sup) else res_.sup[0]
                
                dom_i = Interval(dom.inf[i], dom.sup[i])
                if res_.inf < dom_i.inf:
                    mins[i] = dom_i.inf if np.isscalar(dom_i.inf) else dom_i.inf[0]
                
                if res_.inf > dom_i.sup:
                    maxs[i] = dom_i.sup if np.isscalar(dom_i.sup) else dom_i.sup[0]
                
                dom = Interval(mins, maxs) & dom
                
                continue
            
            # Contract overall domain
            res_ = _aux_contractBox(vars_sym[i], p[j], dom, i, vars_sym)
            mins[i] = min(res_)
            maxs[i] = max(res_)
            dom = Interval(mins, maxs) & dom
    
    return dom


def _aux_contractBox(var: Any, poly: Any, dom: Interval, index: int, vars_sym: List) -> np.ndarray:
    """
    Contract domain based on extremal functions
    
    Args:
        var: symbolic variable for the variable being contracted
        poly: symbolic polynomial expression
        dom: domain (interval object)
        index: index of variable to contract
        vars_sym: list of all symbolic variables
        
    Returns:
        res_: bounds [l; r] for the contracted variable
    """
    import sympy as sp
    
    # Get polynomial coefficients
    poly_obj = sp.Poly(poly, var)
    c = poly_obj.all_coeffs()
    
    # Reverse to match MATLAB order (highest degree first in MATLAB, but sympy gives lowest first)
    c = list(reversed(c))
    
    # Degree of the polynomial
    d = len(c)
    
    # Determine the extremal functions
    infs = np.zeros(d)
    sups = np.zeros(d)
    
    n = dom.dim()
    
    dom_index_sup = dom.sup[index] if dom.sup.size > index else dom.sup
    if dom_index_sup <= 0:
        negInt = 1
    else:
        negInt = 0
    
    # Determine extrema for each coefficient function
    for k in range(d):
        # If the coefficient is 0, just skip the step
        if c[k] == 0:
            continue
        
        # Convert coefficient to function
        # MATLAB: g = matlabFunction(c(k),'Vars',{x});
        g_expr = c[k]
        # lambdify with vars_sym (not wrapped in list) to accept multiple arguments
        g = sp.lambdify(vars_sym, g_expr, 'numpy')
        
        # Evaluate on interval domain
        # MATLAB: if isa(g(dom),'double') == 1
        # MATLAB automatically evaluates g(dom) on interval and returns double if constant
        # In Python, we evaluate at corners to get interval bounds
        dom_inf = dom.inf
        dom_sup = dom.sup
        
        # Generate all corner combinations (2^n points for n dimensions)
        corners = []
        for dim in range(n):
            inf_val = dom_inf[dim] if dom_inf.size > dim else dom_inf
            sup_val = dom_sup[dim] if dom_sup.size > dim else dom_sup
            corners.append([inf_val, sup_val])
        
        # Evaluate at all corners
        g_vals = []
        for combo in itertools.product(*corners):
            val = g(*combo)
            if isinstance(val, (int, float, np.number)):
                g_vals.append(float(val))
            elif isinstance(val, np.ndarray):
                g_vals.extend(val.flatten().tolist())
        
        # Also evaluate at center
        dom_center = dom.center()
        val = g(*dom_center.flatten() if hasattr(dom_center, 'flatten') else dom_center)
        if isinstance(val, (int, float, np.number)):
            g_vals.append(float(val))
        elif isinstance(val, np.ndarray):
            g_vals.extend(val.flatten().tolist())
        
        # Get bounds
        g_inf = min(g_vals)
        g_sup = max(g_vals)
        
        # MATLAB: if isa(g(dom),'double') == 1
        # Check if result is constant (numeric)
        if abs(g_sup - g_inf) < np.finfo(float).eps:
            infs[k] = g_inf
            sups[k] = g_inf
            continue
        
        # If negative value is to be plugged in and the exponent is odd
        if (d % 2 == 1 and k % 2 == 0 and negInt == 1) or \
           (d % 2 == 0 and k % 2 == 1 and negInt == 1):
            infs[k] = g_sup
            sups[k] = g_inf
            continue
        
        infs[k] = g_inf
        sups[k] = g_sup
    
    # Create polynomial functions from coefficients
    # poly2sym equivalent: create polynomial from coefficients
    inf_poly = np.poly1d(infs[::-1])  # Reverse for poly1d (highest degree first)
    sup_poly = np.poly1d(sups[::-1])
    
    infHan = lambda x: inf_poly(x)
    supHan = lambda x: sup_poly(x)
    
    # Determine the left bound of new interval
    dom_index_inf = dom.inf[index] if dom.inf.size > index else dom.inf
    dom_index_sup = dom.sup[index] if dom.sup.size > index else dom.sup
    
    # Initialize l = left bound
    l = dom_index_inf if np.isscalar(dom_index_inf) else dom_index_inf[0]
    
    inf_val_at_inf = inf_poly(l)
    sup_val_at_inf = sup_poly(l)
    
    if inf_val_at_inf <= 0 and sup_val_at_inf >= 0:
        l = dom_index_inf if np.isscalar(dom_index_inf) else dom_index_inf[0]
    
    # inf(g[Y])(inf(x)) > 0
    if inf_val_at_inf > 0:
        # Select inf(g[Y]) for contraction
        if d < 4:
            # Use explicit analytical expressions
            # MATLAB: roots(infs) - note: infs is coefficient vector (highest degree first)
            # infs is already in descending order (highest degree first) after the reverse at line 170
            roots_inf = np.roots(infs)  # np.roots expects highest degree first
            real_roots = roots_inf[np.isreal(roots_inf)]
            if len(real_roots) > 0:
                real_roots = np.real(real_roots)
                valid_roots = real_roots[real_roots > l]
                if len(valid_roots) > 0:
                    l = float(np.min(valid_roots))
        else:
            # Use forward-backward contractor to find zero crossings
            dom_i = Interval(dom_index_inf, dom_index_sup)
            int_contract = contractForwardBackward(infHan, dom_i)
            
            if int_contract is not None:
                l = int_contract.inf if np.isscalar(int_contract.inf) else int_contract.inf[0]
    
    # sup(g[Y])(inf(x)) < 0
    if sup_val_at_inf < 0:
        # Select sup(g[Y]) for contraction
        if d < 4:
            # sups is already in descending order (highest degree first)
            roots_sup = np.roots(sups)
            real_roots = roots_sup[np.isreal(roots_sup)]
            if len(real_roots) > 0:
                real_roots = np.real(real_roots)
                valid_roots = real_roots[real_roots > l]
                if len(valid_roots) > 0:
                    l = float(np.min(valid_roots))
        else:
            # Use forward-backward contractor
            dom_i = Interval(dom_index_inf, dom_index_sup)
            int_contract = contractForwardBackward(supHan, dom_i)
            
            if int_contract is not None:
                l = int_contract.inf if np.isscalar(int_contract.inf) else int_contract.inf[0]
    
    # Determine the right bound of new interval
    # Initialize r = right bound
    r = dom_index_sup if np.isscalar(dom_index_sup) else dom_index_sup[0]
    
    inf_val_at_sup = inf_poly(r)
    sup_val_at_sup = sup_poly(r)
    
    if inf_val_at_sup <= 0 and sup_val_at_sup >= 0:
        r = dom_index_sup if np.isscalar(dom_index_sup) else dom_index_sup[0]
    
    # inf(g[Y])(sup(x)) > 0
    elif inf_val_at_sup > 0:
        # Select inf(g[Y]) for contraction
        if d < 4:
            # MATLAB: r_ = roots(infs); if ~isempty(r_) && isreal(r_) && max(r_) < r
            # infs is already in descending order (highest degree first)
            roots_inf = np.roots(infs)
            real_roots = roots_inf[np.isreal(roots_inf)]
            if len(real_roots) > 0:
                real_roots = np.real(real_roots)
                valid_roots = real_roots[real_roots < r]
                if len(valid_roots) > 0:
                    r = float(np.max(valid_roots))
        else:
            # Use forward-backward contractor
            dom_i = Interval(dom_index_inf, dom_index_sup)
            int_contract = contractForwardBackward(infHan, dom_i)
            
            if int_contract is not None:
                r = int_contract.sup if np.isscalar(int_contract.sup) else int_contract.sup[0]
    
    # sup(g[Y])(sup(x)) < 0
    elif sup_val_at_sup < 0:
        # Select sup(g[Y]) for contraction
        if d < 4:
            # sups is already in descending order (highest degree first)
            roots_sup = np.roots(sups)
            real_roots = roots_sup[np.isreal(roots_sup)]
            if len(real_roots) > 0:
                real_roots = np.real(real_roots)
                valid_roots = real_roots[real_roots < r]
                if len(valid_roots) > 0:
                    r = float(np.max(valid_roots))
        else:
            # Use forward-backward contractor
            dom_i = Interval(dom_index_inf, dom_index_sup)
            int_contract = contractForwardBackward(supHan, dom_i)
            
            if int_contract is not None:
                r = int_contract.sup if np.isscalar(int_contract.sup) else int_contract.sup[0]
    
    res_ = np.array([l, r])
    return res_

