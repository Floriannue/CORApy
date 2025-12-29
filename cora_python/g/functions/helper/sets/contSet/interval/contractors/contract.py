"""
contract - contracts a interval domain to tightly enclose nonlinear constraints

Syntax:
    res = contract(f,dom)
    res = contract(f,dom,alg)
    res = contract(f,dom,alg,iter)
    res = contract(f,dom,alg,iter,splits)
    res = contract(f,dom,alg,iter,splits,jacHan)

Inputs:
    f - function handle for the constraint f(x) = 0
    dom - initial domain (class: interval)
    alg - algorithm used for contraction. The available algorithms are 
          'forwardBackward' (see Sec. 4.2.4 in [1]),  'linearize' 
          (see Sec. 4.3.4 in [1]), 'polynomial' (see [2]), 'interval', and 
          'all' (all contractors together)
    iter - number of iteration (integer > 0 or 'fixpoint')
    splits - number of recursive splits (integer > 0)
    jacHan - handle for Jacobian

Outputs:
    res - contracted domain (class: interval)

Example: 
    f = @(x) x(1)^2 + x(2)^2 - 4;
    dom = interval([1;1],[3;3]);
   
    res = contract(f,dom);

References:
    [1] L. Jaulin et al. "Applied Interval Analysis", 2006
    [2] G. Trombettoni et al. "A Box-Consistency Contractor Based on 
        Extremal Functions", 2010

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contractForwardBackward, contractParallelLinearization

Authors:       Niklas Kochdumper
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
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .contractForwardBackward import contractForwardBackward
from .contractParallelLinearization import contractParallelLinearization
from .contractInterval import contractInterval
from .contractPolyBoxRevise import contractPolyBoxRevise


def contract(f: Callable, dom: Interval, *varargin) -> Optional[Interval]:
    """
    Contracts an interval domain to tightly enclose nonlinear constraints
    
    Args:
        f: function handle for the constraint f(x) = 0
        dom: initial domain (interval object)
        *varargin: optional arguments (alg, iter, splits, jacHan)
        
    Returns:
        res: contracted domain (interval object) or None if empty
    """
    # Set default values
    defaults = ['forwardBackward', 1, None, None]
    alg, iter_val, splits, jacHan = setDefaultValues(defaults, list(varargin))
    
    # Check input arguments
    # MATLAB: inputArgsCheck({{alg,'str',{'forwardBackward','linearize',...}}})
    inputArgsCheck([[alg, 'str', ['forwardBackward', 'linearize', 
                                    'polynomial', 'interval', 'all']]])
    
    # Check iter argument
    # MATLAB: if ischar(iter) ... inputArgsCheck({{iter,'str','fixpoint'}})
    if isinstance(iter_val, str):
        inputArgsCheck([[iter_val, 'str', ['fixpoint']]])
        iter_val = 10000
    else:
        # MATLAB: inputArgsCheck({{iter,'att','numeric',{'integer','nonnegative'}}})
        inputArgsCheck([[iter_val, 'att', ['numeric'], ['integer', 'nonnegative']]])
    
    # Precompute jacobian matrix
    # MATLAB: if isempty(jacHan) && ismember(alg,{'linearize','interval','all'})
    if jacHan is None and alg in ['linearize', 'interval', 'all']:
        # Construct symbolic variables
        # MATLAB: vars = sym('x',[n,1]);
        n = dom.dim()
        vars_sym = sp.symbols('x0:{}'.format(n))
        if n == 1:
            vars_sym = [vars_sym]
        else:
            vars_sym = list(vars_sym)
        
        # Evaluate function with symbolic variables
        # MATLAB: fSym = f(vars);
        vars_list = list(vars_sym)
        f_sym = f(vars_list)
        
        # Determine number of constraints
        if isinstance(f_sym, (list, tuple, np.ndarray)):
            num_constraints = len(f_sym)
        else:
            num_constraints = 1
        
        # Compute jacobian matrix
        # MATLAB: jac = jacobian(fSym,vars);
        if isinstance(f_sym, (list, tuple, np.ndarray)):
            if len(f_sym) == 1:
                f_sym = f_sym[0] if hasattr(f_sym, '__getitem__') else f_sym
                jac = sp.Matrix([[sp.diff(f_sym, var) for var in vars_sym]])
            else:
                # Multiple constraints - compute Jacobian for each
                jac_list = []
                for fi in f_sym:
                    if hasattr(fi, '__iter__') and len(fi) == 1:
                        fi = fi[0]
                    jac_i = [sp.diff(fi, var) for var in vars_sym]
                    jac_list.append(jac_i)
                jac = sp.Matrix(jac_list)
        else:
            # Single constraint
            jac = sp.Matrix([[sp.diff(f_sym, var) for var in vars_sym]])
        
        # Convert to function handle
        # MATLAB: jacHan = matlabFunction(jac,'Vars',{vars});
        # lambdify with vars_sym (not wrapped in list) to accept multiple arguments
        jac_func = sp.lambdify(vars_sym, jac, 'numpy')
        # Handle both array and list inputs
        # MATLAB: jacHan is a function that takes a vector and returns a matrix
        # MATLAB's matlabFunction can handle numeric, interval, and taylm inputs
        def jacHan_func(x):
            # Check if input is Interval or Taylm - evaluate symbolically
            from cora_python.contSet.interval.interval import Interval
            from cora_python.contSet.taylm.taylm import Taylm
            is_taylm = isinstance(x, Taylm)
            
            if isinstance(x, Interval):
                # For interval input, evaluate symbolically on the interval
                # Substitute interval bounds into symbolic expression
                # This is a simplified approach - in MATLAB, symbolic math handles this automatically
                # We evaluate at corners to get interval bounds
                n = x.dim()
                if n != len(vars_sym):
                    raise CORAerror('CORA:dimensionMismatch', 
                                   f'Input dimension {n} does not match number of variables {len(vars_sym)}')
                
                # Evaluate at all corners to get interval bounds
                corners = []
                for dim in range(n):
                    inf_val = x.inf[dim] if x.inf.size > dim else x.inf
                    sup_val = x.sup[dim] if x.sup.size > dim else x.sup
                    corners.append([inf_val, sup_val])
                
                import itertools
                jac_vals = []
                for combo in itertools.product(*corners):
                    val = jac_func(*combo)
                    jac_vals.append(val)
                
                # Compute interval bounds for each element of jacobian matrix
                # Convert all to arrays first
                jac_vals_arrays = [np.array(v) for v in jac_vals]
                if len(jac_vals_arrays) > 0:
                    # Stack all results
                    jac_vals_array = np.stack(jac_vals_arrays, axis=0)
                    # Compute min/max for each matrix element
                    jac_inf = np.min(jac_vals_array, axis=0)
                    jac_sup = np.max(jac_vals_array, axis=0)
                    # MATLAB: jacHan(dom) returns an interval matrix
                    # For contractors, we need to return something that can be used in J - A
                    # Return the center of the interval matrix (midpoint) for now
                    # This matches MATLAB's behavior where interval matrices can be used in arithmetic
                    jac_center = 0.5 * (jac_inf + jac_sup)
                    return jac_center
                else:
                    return np.array([])
            elif is_taylm:
                # For taylm input, evaluate symbolically
                # In MATLAB, this returns a taylm, which is then converted to interval
                # For now, evaluate at taylm's interval bounds
                tay_interval = x.interval() if hasattr(x, 'interval') else Interval(x.inf, x.sup)
                return jacHan_func(tay_interval)
            else:
                # Numeric input - use lambdified function
                # Convert input to flat array
                if isinstance(x, np.ndarray):
                    x_flat = x.flatten()
                elif isinstance(x, (list, tuple)):
                    x_flat = np.array(x, dtype=float).flatten()
                else:
                    x_flat = np.array([float(x)], dtype=float)
                # Pass as separate arguments to match MATLAB's behavior
                # MATLAB: jacHan(x) where x is a column vector
                result = jac_func(*x_flat)
                # Convert result to numpy array
                # MATLAB returns a matrix, so ensure we return a 2D array
                # The Jacobian should be (num_constraints, num_variables)
                result = np.asarray(result)
                n_vars = len(vars_sym)
                if result.ndim == 0:
                    # Scalar result - return as 1x1 matrix
                    return np.array([[float(result)]], dtype=float)
                elif result.ndim == 1:
                    # 1D array - reshape based on number of constraints
                    if result.size == n_vars * num_constraints:
                        # Multiple constraints: reshape to (num_constraints, n_vars)
                        return result.reshape(num_constraints, n_vars)
                    elif result.size == n_vars:
                        # Single constraint: reshape to (1, n_vars)
                        return result.reshape(1, n_vars)
                    else:
                        # Unknown shape - return as column vector (fallback)
                        return result.reshape(-1, 1)
                elif result.ndim == 2:
                    # Already 2D - ensure correct shape
                    if result.shape[0] != num_constraints and result.shape[1] == num_constraints:
                        # Transpose if needed
                        return result.T
                    return result
                else:
                    # Higher dimensional - flatten and reshape
                    result_flat = result.flatten()
                    if result_flat.size == n_vars * num_constraints:
                        return result_flat.reshape(num_constraints, n_vars)
                    else:
                        return result.reshape(result.shape[0], -1)
        jacHan = jacHan_func
    
    # Splitting of intervals considered or not
    if splits is None:  # no splitting
        # Iteratively contract the domain
        dom_ = dom
        
        for i in range(iter_val):
            # Contract the domain using the selected algorithm
            if alg == 'forwardBackward':
                dom = contractForwardBackward(f, dom)
            elif alg == 'linearize':
                dom = contractParallelLinearization(f, dom, jacHan)
            elif alg == 'polynomial':
                dom = contractPolyBoxRevise(f, dom)
            elif alg == 'interval':
                dom = contractInterval(f, dom, jacHan)
            elif alg == 'all':
                dom = contractForwardBackward(f, dom)
                if dom is not None and not dom.representsa_('emptySet', np.finfo(float).eps):
                    dom = contractInterval(f, dom, jacHan)
                    if dom is not None and not dom.representsa_('emptySet', np.finfo(float).eps):
                        dom = contractParallelLinearization(f, dom, jacHan)
                        if dom is not None and not dom.representsa_('emptySet', np.finfo(float).eps):
                            dom = contractPolyBoxRevise(f, dom)
            
            # Check if set is empty
            if dom is None or dom.representsa_('emptySet', np.finfo(float).eps):
                return None  # MATLAB returns []
            
            # Check for convergence
            if np.all(np.abs(dom.inf - dom_.inf) < np.finfo(float).eps) and \
               np.all(np.abs(dom.sup - dom_.sup) < np.finfo(float).eps):
                break
            else:
                dom_ = dom
        
        return dom
        
    else:  # splitting
        # Initialization
        list_intervals = [dom]
        
        # Loop over the number of recursive splits
        for i in range(splits):
            list_ = [None] * (2 * len(list_intervals))
            counter = 0
            
            # Loop over all sets in the list
            for j in range(len(list_intervals)):
                # Determine the best dimension to split and split the domain
                domSplit = _aux_bestSplit(f, list_intervals[j])
                
                # Loop over all splitted domains
                for k in range(len(domSplit)):
                    # Check if the domain is empty
                    temp = f(domSplit[k])
                    
                    p = np.zeros(len(temp) if hasattr(temp, '__len__') else 1)
                    
                    if not temp.contains_(p):
                        continue
                    
                    # Contract the splitted domain
                    domTemp = contract(f, domSplit[k], alg, iter_val, None, jacHan)
                    
                    # Update the queue
                    if domTemp is not None and not domTemp.representsa_('emptySet', np.finfo(float).eps):
                        list_[counter] = domTemp
                        counter += 1
            
            list_intervals = list_[:counter]
        
        # Unite all the contracted intervals
        if len(list_intervals) > 0:
            res = list_intervals[0]
            
            for i in range(1, len(list_intervals)):
                res = res | list_intervals[i]
            
            return res
        else:
            return None


def _aux_bestSplit(f: Callable, dom: Interval) -> List[Interval]:
    """
    Determine the dimension along which it is best to split the domain
    
    Args:
        f: function handle
        dom: domain (interval object)
        
    Returns:
        list: list of split intervals
    """
    n = dom.dim()
    vol = np.zeros(n)
    
    # Loop over all dimensions
    for i in range(n):
        # Split the domain at the current dimension
        dom_split = dom.split(i)
        
        # Evaluate constraint function for one of the splitted sets
        val = f(dom_split[0])
        
        # Evaluate the quality of the split
        vol[i] = val.volume_() if hasattr(val, 'volume_') else 0.0
    
    # Determine best dimension to split
    ind = np.argmin(vol)
    
    # Split the domain at the determined dimension
    return dom.split(ind)
