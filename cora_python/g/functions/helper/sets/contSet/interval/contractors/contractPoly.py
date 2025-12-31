"""
contractPoly - contracts a interval domain to tightly enclose polynomial
               constraints

Description:
   Contract an interval domain for polynomial constraints defined as:
   f(x) = c + sum{i=1}prod{k=1} x(k)^E(k,i) + sum{j=1} x(j)GI(j,:) = 0

Syntax:
   res = contractPoly(c,G,GI,E,dom)
   res = contractPoly(c,G,GI,E,dom,alg)
   res = contractPoly(c,G,GI,E,dom,alg,iter)
   res = contractPoly(c,G,GI,E,dom,alg,iter,splits)
   res = contractPoly(c,G,GI,E,dom,alg,iter,splits,jacHan)

Inputs:
   c - constant offset of the polynomial constraint
   G - matrix of dependent generators for the polynomial constraint
   GI - matrix of independent generators for the polynomial constraint
   E - exponent matrix for the polynomial constraint
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
from typing import TYPE_CHECKING, Optional, Callable, Union, List, Any

from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
from cora_python.specification.syntaxTree import syntaxTree, SyntaxTree
from .contractForwardBackward import contractForwardBackward
from .contractParallelLinearization import contractParallelLinearization
from .contractInterval import contractInterval


def contractPoly(c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray,
                 dom: 'Interval', alg: Optional[str] = None,
                 iter_val: Union[int, str] = 1, splits: Optional[int] = None,
                 jacHan: Optional[Callable] = None) -> Optional['Interval']:
    """
    Contracts an interval domain to tightly enclose polynomial constraints
    
    Args:
        c: constant offset of the polynomial constraint
        G: matrix of dependent generators for the polynomial constraint
        GI: matrix of independent generators for the polynomial constraint
        E: exponent matrix for the polynomial constraint
        dom: initial domain (interval object)
        alg: algorithm used for contraction ('forwardBackward', 'linearize',
             'polynomial', 'interval', or 'all')
        iter_val: number of iteration (integer > 0 or 'fixpoint')
        splits: number of recursive splits (integer > 0)
        jacHan: handle for Jacobian
        
    Returns:
        res: contracted domain (interval object) or None if empty
    """
    
    # Set default values
    defaults = ['forwardBackward', 1, None, None]
    varargin = [alg, iter_val, splits, jacHan]
    alg, iter_val, splits, jacHan = setDefaultValues(defaults, varargin)
    
    # Check input arguments
    # MATLAB: inputArgsCheck({{alg,'str',{'forwardBackward','linearize',...}}})
    inputArgsCheck([[alg, 'str', ['forwardBackward', 'linearize', 
                                    'polynomial', 'interval', 'all']]])
    
    # Check iter_val
    # MATLAB: if ischar(iter) ... inputArgsCheck({{iter,'str','fixpoint'}})
    if isinstance(iter_val, str):
        inputArgsCheck([[iter_val, 'str', ['fixpoint']]])
        iter_val = 10000
    else:
        # MATLAB: inputArgsCheck({{iter,'att','numeric',{'integer','nonnegative'}}})
        inputArgsCheck([[iter_val, 'att', ['numeric'], ['integer', 'nonnegative']]])
    
    # Function handle for polynomial constrained function
    def f(x):
        return _aux_funcPoly(x, c, G, GI, E)
    
    # Precompute jacobian matrix
    if (jacHan is None or (hasattr(jacHan, '__len__') and len(jacHan) == 0)) and \
       alg in ['linearize', 'interval', 'all']:
        
        # Compute exponent matrix differentiated for each variable
        Elist = []
        Glist = []
        
        for i in range(E.shape[0]):
            ind = np.where(E[i, :] > 0)[0]
            if len(ind) > 0:
                temp = E[:, ind].copy()
                temp[i, :] = temp[i, :] - 1
                Elist.append(temp)
                Glist.append(G[:, ind] @ np.diag(E[i, ind]))
            else:
                Elist.append(np.array([]).reshape(E.shape[0], 0))
                Glist.append(np.array([]).reshape(G.shape[0], 0))
        
        # Function handle for jacobian matrix
        def jacHan_func(x):
            return _aux_jacobianPoly(x, Glist, GI, Elist, G.shape[0], E.shape[0])
        
        jacHan = jacHan_func
    
    # Splitting of intervals considered or not
    if splits is None or (hasattr(splits, '__len__') and len(splits) == 0):  # No splitting
        
        # Copy domain to avoid in-place modification
        dom = Interval(dom)
        
        # Iteratively contract the domain
        dom_ = dom
        
        for i in range(iter_val):
            # Contract the domain using the selected algorithm
            if alg == 'forwardBackward':
                dom = contractForwardBackward(f, dom)
            elif alg == 'linearize':
                dom = contractParallelLinearization(f, dom, jacHan, 'interval')
            elif alg == 'polynomial':
                dom = _aux_contractPolyBoxRevisePoly(c, G, GI, E, dom)
            elif alg == 'interval':
                dom = contractInterval(f, dom, jacHan, 'interval')
            elif alg == 'all':
                # Check if domain is a point interval
                # If it is, skip contractForwardBackward, contractInterval, and contractParallelLinearization
                # (which all return None for points that don't satisfy constraint exactly)
                # and go directly to _aux_contractPolyBoxRevisePoly (which accepts point intervals)
                width = dom.sup - dom.inf
                is_point = np.all(np.abs(width) < np.finfo(float).eps)
                
                if is_point:
                    # For point intervals, skip to _aux_contractPolyBoxRevisePoly directly
                    dom = _aux_contractPolyBoxRevisePoly(c, G, GI, E, dom)
                else:
                    dom = contractForwardBackward(f, dom)
                    if dom is not None and not dom.representsa_('emptySet', np.finfo(float).eps):
                        dom = contractInterval(f, dom, jacHan, 'interval')
                        if dom is not None and not dom.representsa_('emptySet', np.finfo(float).eps):
                            dom = contractParallelLinearization(f, dom, jacHan, 'interval')
                            if dom is not None and not dom.representsa_('emptySet', np.finfo(float).eps):
                                dom = _aux_contractPolyBoxRevisePoly(c, G, GI, E, dom)
            
            # Check if set is empty or None
            if dom is None:
                return None  # MATLAB returns []
            if dom.representsa_('emptySet', np.finfo(float).eps):
                return None  # MATLAB returns []
            
            # Check for convergence
            if np.all(np.abs(dom.inf - dom_.inf) < np.finfo(float).eps) and \
               np.all(np.abs(dom.sup - dom_.sup) < np.finfo(float).eps):
                break
            else:
                dom_ = dom
        
        return dom
        
    else:  # Splitting
        # Initialization
        # Copy domain to avoid in-place modification
        list_intervals = [Interval(dom)]
        
        # Loop over the number of recursive splits
        for i in range(splits):
            list_new = [None] * (2 * len(list_intervals))
            counter = 0
            
            # Loop over all sets in the list
            for j in range(len(list_intervals)):
                # Determine the best dimension to split and split the domain
                domSplit = _aux_bestSplit(f, list_intervals[j])
                
                # Loop over all splitted domains
                for k in range(len(domSplit)):
                    # Check if the domain is empty
                    temp = f(domSplit[k])
                    
                    p = np.zeros(len(temp))
                    
                    if not temp.contains_(p):
                        continue
                    
                    # Contract the splitted domain
                    domTemp = contractPoly(c, G, GI, E, domSplit[k], alg, iter_val, None, jacHan)
                    
                    # Update the queue
                    if domTemp is not None and not domTemp.representsa_('emptySet', np.finfo(float).eps):
                        list_new[counter] = domTemp
                        counter += 1
            
            list_intervals = list_new[:counter]
        
        # Unite all the contracted intervals
        if len(list_intervals) > 0:
            res = list_intervals[0]
            
            for i in range(1, len(list_intervals)):
                res = res | list_intervals[i]
            
            return res
        else:
            return None  # MATLAB returns []


# Auxiliary functions -----------------------------------------------------

def _aux_jacobianPoly(x: Any, G: List[np.ndarray], GI: np.ndarray, E: List[np.ndarray],
                      n: int, p: int) -> np.ndarray:
    """
    Compute the jacobian matrix for the polynomial constraint
    
    Args:
        x: input (can be interval, taylm, or numeric)
        G: list of generator matrices for each variable
        GI: independent generator matrix
        E: list of exponent matrices for each variable
        n: number of constraints (rows of G)
        p: number of dependent variables (rows of E)
        
    Returns:
        J: jacobian matrix
    """
    
    # Handle different input types
    if isinstance(x, Interval):
        x_vals = [Interval(x.inf[i], x.sup[i]) for i in range(min(p, x.dim()))]
        if len(x_vals) < p:
            x_vals.extend([Interval(0, 0)] * (p - len(x_vals)))
    elif hasattr(x, '__len__'):
        x_vals = x[:p] if len(x) >= p else list(x) + [0] * (p - len(x))
    else:
        x_vals = [x] * p
    
    # Initialize J
    # MATLAB: J = 0 * repmat(x(1),[n,length(E)]);
    # This creates a matrix of zeros with the same type as x(1)
    if isinstance(x_vals[0], Interval):
        # Use interval arithmetic - initialize with Interval(0,0)
        J = np.array([[Interval(0, 0) for _ in range(len(E))] for _ in range(n)], dtype=object)
        for i in range(len(E)):
            for j in range(E[i].shape[1]):
                # Compute prod(x_.^E{i}(:,j))
                prod_val = Interval(1, 1)
                for k in range(len(x_vals)):
                    if E[i][k, j] > 0:
                        prod_val = prod_val * (x_vals[k] ** E[i][k, j])
                J[:, i] = J[:, i] + G[i][:, j] * prod_val
    else:
        # Numeric computation
        J = np.zeros((n, len(E)))
        x_ = np.asarray(x_vals[:p]).flatten()
        for i in range(len(E)):
            for j in range(E[i].shape[1]):
                # Compute prod(x_.^E{i}(:,j))
                prod_val = np.prod(x_ ** E[i][:, j])
                J[:, i] = J[:, i] + G[i][:, j] * prod_val
    
    # Consider independent generators
    if GI.size > 0:
        J = np.hstack([J, GI])
    
    return J


def _aux_funcPoly(x: Any, c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray) -> Any:
    """
    Evaluate the polynomial function at the point x
    
    Args:
        x: input (can be interval, taylm, or numeric)
        c: constant offset
        G: generator matrix
        GI: independent generator matrix
        E: exponent matrix
        
    Returns:
        f: function value
    """
    
    # #region agent log
    import json
    with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
        c_val = float(c) if np.isscalar(c) else (c.tolist() if hasattr(c, 'tolist') else str(c))
        G_shape = G.shape.tolist() if hasattr(G.shape, 'tolist') else list(G.shape)
        E_shape = E.shape.tolist() if hasattr(E.shape, 'tolist') else list(E.shape)
        log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"contractPoly.py:_aux_funcPoly:entry","message":"_aux_funcPoly called","data":{"x_type":type(x).__name__,"x_len":len(x) if hasattr(x, '__len__') else None,"c":c_val,"G_shape":G_shape,"E_shape":E_shape},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    # Initialization
    n = E.shape[0]
    
    # Handle different input types
    if isinstance(x, Interval):
        x_dim = x.dim()
        x1 = [Interval(x.inf[i], x.sup[i]) for i in range(min(n, x_dim))]
        if len(x1) < n:
            x1.extend([Interval(0, 0)] * (n - len(x1)))
        x2 = [Interval(x.inf[i], x.sup[i]) for i in range(n, x_dim)] if x_dim > n else []
    elif hasattr(x, '__len__'):
        x1 = x[:n] if len(x) >= n else list(x[:len(x)]) + [0] * (n - len(x))
        x2 = x[n:] if len(x) > n else []
    else:
        x1 = [x] * n
        x2 = []
    
    # Check if x1[0] is an Interval or syntax tree node
    is_interval = len(x1) > 0 and isinstance(x1[0], Interval)
    is_syntax_tree = len(x1) > 0 and isinstance(x1[0], SyntaxTree)
    
    # #region agent log
    with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
        log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"contractPoly.py:_aux_funcPoly:type_check","message":"Input type detected","data":{"is_interval":bool(is_interval),"is_syntax_tree":bool(is_syntax_tree),"x1_0_type":type(x1[0]).__name__ if len(x1) > 0 else None},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    # Initialize f based on input type
    if is_interval:
        # Handle c as scalar or array for interval case
        if np.isscalar(c):
            f = Interval(c, c)
        else:
            f = Interval(c, c) if c.ndim == 0 else Interval(c.flatten(), c.flatten())
    elif is_syntax_tree:
        # For syntax tree, convert c to syntax tree(s)
        if np.isscalar(c):
            f = syntaxTree(Interval(c, c), None)
        else:
            # For multiple constraints, create array of syntax trees
            c_flat = c.flatten()
            f = [syntaxTree(Interval(c_flat[j], c_flat[j]), None) for j in range(len(c_flat))]
    else:
        # Numeric case
        if np.isscalar(c):
            f = np.array([c])
        else:
            f = c.copy()
    
    # Loop over all dependent generators
    for i in range(G.shape[1]):
        if is_interval or is_syntax_tree:
            # Interval or syntax tree arithmetic
            if is_interval:
                prod_val = Interval(1, 1)
            else:
                # For syntax tree, start with 1 (as interval)
                prod_val = syntaxTree(Interval(1, 1), None)
            
            for k in range(n):
                if E[k, i] > 0:
                    # #region agent log
                    with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                        log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"contractPoly.py:_aux_funcPoly:before_pow","message":"Before power operation","data":{"k":k,"E_ki":float(E[k, i]),"x1_k_type":type(x1[k]).__name__},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    pow_result = x1[k] ** E[k, i]
                    # #region agent log
                    with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                        pow_type = type(pow_result).__name__
                        pow_operator = getattr(pow_result, 'operator', None) if hasattr(pow_result, 'operator') else None
                        log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"contractPoly.py:_aux_funcPoly:after_pow","message":"After power operation","data":{"pow_type":pow_type,"pow_operator":pow_operator},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    prod_val = prod_val * pow_result
            
            # G[:, i] is a vector, prod_val is an interval scalar or syntax tree
            G_col = G[:, i] if G.ndim > 1 else G
            
            if is_interval:
                # Interval case: G[:, i] * prod_val gives interval vector
                from cora_python.contSet.interval.times import times
                G_prod = times(prod_val, G_col)  # interval * vector gives interval vector
                f = f + G_prod
            else:
                # Syntax tree case: handle each constraint separately
                if isinstance(f, list):
                    # Multiple constraints
                    for j in range(len(f)):
                        G_val = Interval(G_col[j], G_col[j])
                        G_prod = prod_val * G_val
                        f[j] = f[j] + G_prod
                else:
                    # Single constraint
                    G_val = Interval(G_col[0], G_col[0]) if len(G_col) > 0 else Interval(0, 0)
                    G_prod = prod_val * G_val
                    f = f + G_prod
        else:
            # Numeric computation
            x1_arr = np.asarray(x1[:n]).flatten()
            prod_val = np.prod(x1_arr ** E[:, i])
            f = f + G[:, i] * prod_val
    
    # Add independent generators
    if GI.size > 0 and len(x2) > 0:
        if is_interval:
            f = f + GI @ x2
        elif is_syntax_tree:
            # For syntax tree, handle GI @ x2
            if isinstance(f, list):
                # Multiple constraints
                for j in range(len(f)):
                    GI_row = GI[j, :] if GI.ndim > 1 else GI
                    GI_sum = syntaxTree(Interval(0, 0), None)
                    for k in range(len(x2)):
                        GI_val = Interval(GI_row[k], GI_row[k]) if np.isscalar(GI_row[k]) else Interval(GI_row[k], GI_row[k])
                        GI_sum = GI_sum + (x2[k] * GI_val)
                    f[j] = f[j] + GI_sum
            else:
                # Single constraint
                GI_row = GI[0, :] if GI.ndim > 1 else GI
                GI_sum = syntaxTree(Interval(0, 0), None)
                for k in range(len(x2)):
                    GI_val = Interval(GI_row[k], GI_row[k]) if np.isscalar(GI_row[k]) else Interval(GI_row[k], GI_row[k])
                    GI_sum = GI_sum + (x2[k] * GI_val)
                f = f + GI_sum
        else:
            # Numeric case
            x2_arr = np.asarray(x2).flatten().reshape(-1, 1)
            f = f + GI @ x2_arr
    
    # #region agent log
    with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
        f_type = type(f).__name__
        f_info = None
        if isinstance(f, SyntaxTree):
            f_info = {"type":"SyntaxTree","has_value":hasattr(f, 'value'),"operator":getattr(f, 'operator', None)}
        elif isinstance(f, list):
            f_info = {"type":"list","len":len(f),"first_type":type(f[0]).__name__ if len(f) > 0 else None}
        elif isinstance(f, Interval):
            f_info = {"type":"Interval","inf":f.inf.tolist() if hasattr(f.inf, 'tolist') else str(f.inf),"sup":f.sup.tolist() if hasattr(f.sup, 'tolist') else str(f.sup)}
        log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"contractPoly.py:_aux_funcPoly:return","message":"_aux_funcPoly returning","data":{"f_type":f_type,"f_info":f_info},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    return f


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
        vol[i] = val.volume_()
    
    # Determine best dimension to split
    ind = np.argmin(vol)
    
    # Split the domain at the determined dimension
    return dom.split(ind)


def _aux_contractPolyBoxRevisePoly(c: np.ndarray, G: np.ndarray, GI: np.ndarray,
                                    E: np.ndarray, dom: 'Interval') -> Optional['Interval']:
    """
    Implementation of the contractor based on extremal functions for polynomial constraints
    
    Args:
        c: constant offset
        G: generator matrix
        GI: independent generator matrix
        E: exponent matrix
        dom: domain (interval object)
        
    Returns:
        res: contracted domain (interval object) or None if empty
    """
    from cora_python.contSet.interval.interval import Interval
    from cora_python.contSet.interval.infimum import infimum
    from cora_python.contSet.interval.supremum import supremum
    from cora_python.contSet.interval.representsa_ import representsa_
    
    # Initialization
    n = dom.dim()
    res = dom
    
    # Execute contraction for each variable
    for i in range(n):
        # Loop over all constraints
        # c can be scalar or array
        if np.isscalar(c):
            c_len = 1
            c_val = c
        else:
            c_len = len(c) if hasattr(c, '__len__') else 1
            c_val = c[j] if c_len > 1 else c[0] if hasattr(c, '__getitem__') else c
        
        for j in range(c_len):
            if not np.isscalar(c):
                c_val = c[j] if hasattr(c, '__getitem__') else c
            # Test if a split is needed
            # Access i-th dimension of interval
            dom_i_inf = res.inf[i] if res.inf.size > i else res.inf
            dom_i_sup = res.sup[i] if res.sup.size > i else res.sup
            
            if dom_i_inf < 0 and dom_i_sup > 0:
                # Contract first splitted domain
                tmp_inf = res.inf.copy()
                tmp_sup = res.sup.copy()
                int1 = Interval(dom_i_inf, 0)
                new_inf = res.inf.copy()
                new_sup = res.sup.copy()
                new_inf[i] = dom_i_inf
                new_sup[i] = 0
                res = Interval(new_inf, new_sup)
                
                if GI.size > 0:
                    G_row = G[j, :] if G.ndim > 1 else G
                    GI_row = GI[j, :] if GI.ndim > 1 else GI
                    res1_ = _aux_contractBox(c_val, G_row, GI_row, E, res, i)
                else:
                    G_row = G[j, :] if G.ndim > 1 else G
                    res1_ = _aux_contractBox(c_val, G_row, np.array([]), E, res, i)
                
                # Contract second splitted domain
                int2 = Interval(0, dom_i_sup)
                if hasattr(res, '__setitem__'):
                    res[i] = int2
                else:
                    new_inf = res.inf.copy()
                    new_sup = res.sup.copy()
                    new_inf[i] = 0
                    new_sup[i] = dom_i_sup
                    res = Interval(new_inf, new_sup)
                
                if GI.size > 0:
                    G_row = G[j, :] if G.ndim > 1 else G
                    GI_row = GI[j, :] if GI.ndim > 1 else GI
                    res2_ = _aux_contractBox(c_val, G_row, GI_row, E, res, i)
                else:
                    G_row = G[j, :] if G.ndim > 1 else G
                    res2_ = _aux_contractBox(c_val, G_row, np.array([]), E, res, i)
                
                # Restore original domain
                res = Interval(tmp_inf, tmp_sup)
                
                # Combine the results
                res_ = Interval(min(res1_), max(res2_))
                dom_i = Interval(res.inf[i], res.sup[i])
                temp = dom_i & res_
                
                if not temp.representsa_('emptySet', np.finfo(float).eps):
                    new_inf = res.inf.copy()
                    new_sup = res.sup.copy()
                    new_inf[i] = temp.inf if np.isscalar(temp.inf) else temp.inf[0]
                    new_sup[i] = temp.sup if np.isscalar(temp.sup) else temp.sup[0]
                    res = Interval(new_inf, new_sup)
                else:
                    return None  # MATLAB returns []
                
                continue
            
            # Contract overall domain
            if GI.size > 0:
                G_row = G[j, :] if G.ndim > 1 else G
                GI_row = GI[j, :] if GI.ndim > 1 else GI
                res_ = _aux_contractBox(c_val, G_row, GI_row, E, res, i)
            else:
                G_row = G[j, :] if G.ndim > 1 else G
                res_ = _aux_contractBox(c_val, G_row, np.array([]), E, res, i)
            
            dom_i = Interval(res.inf[i], res.sup[i])
            temp = dom_i & Interval(min(res_), max(res_))
            
            if not temp.representsa_('emptySet', np.finfo(float).eps):
                new_inf = res.inf.copy()
                new_sup = res.sup.copy()
                new_inf[i] = temp.inf if np.isscalar(temp.inf) else temp.inf[0]
                new_sup[i] = temp.sup if np.isscalar(temp.sup) else temp.sup[0]
                res = Interval(new_inf, new_sup)
            else:
                return None  # MATLAB returns []
    
    return res


def _aux_contractBox(c: float, G: np.ndarray, GI: np.ndarray, E: np.ndarray,
                     dom: Interval, index: int) -> np.ndarray:
    """
    Contract domain based on extremal functions
    
    Args:
        c: constant value
        G: generator row vector
        GI: independent generator row vector
        E: exponent matrix
        dom: domain (interval object)
        index: index of variable to contract
        
    Returns:
        res_: bounds [l; r] for the contracted variable
    """
    from .contractForwardBackward import contractForwardBackward
    
    # Independent vs. dependent variable
    if index <= E.shape[0]:  # Dependent variable
        # Degree of the polynomial
        d = int(np.max(E[index, :])) if E.size > 0 else 0
        
        # Initialization
        infs = np.zeros(d + 1)
        sups = np.zeros(d + 1)
        
        infs[0] = c
        sups[0] = c
        
        dom_index_inf = dom.inf[index] if dom.inf.size > index else dom.inf
        dom_index_sup = dom.sup[index] if dom.sup.size > index else dom.sup
        
        if dom_index_sup <= 0:
            negInt = 1
        else:
            negInt = 0
        
        # Determine extrema for each coefficient function
        for k in range(d + 1):
            # Find polynomial expression for the current coefficient
            ind = np.where(E[index, :] == k)[0]
            
            # If the coefficient is 0, just skip the step
            if len(ind) == 0:
                continue
            
            # Compute interval range for the coefficient
            E_ = E[:, ind].copy()
            E_[index, :] = 0
            G_ = G[:, ind] if G.ndim > 1 else G[ind]
            
            # Create zero constant with same shape as c
            c_zero = np.array([0.0]) if np.isscalar(c) else np.zeros_like(c)
            g = _aux_funcPoly(dom, c_zero, G_.reshape(-1, 1) if G_.ndim == 1 else G_,
                             np.array([]), E_)
            
            if isinstance(g, (int, float, np.number)) or (isinstance(g, np.ndarray) and g.size == 1):
                infs[k] = float(g)
                sups[k] = float(g)
                continue
            
            # If negative value is to be plugged in and the exponent is odd
            if k % 2 == 1 and negInt == 1:
                if isinstance(g, Interval):
                    infs[k] = infs[k] + g.sup
                    sups[k] = sups[k] + g.inf
                else:
                    infs[k] = infs[k] + float(np.max(g))
                    sups[k] = sups[k] + float(np.min(g))
                continue
            
            if isinstance(g, Interval):
                infs[k] = infs[k] + g.inf
                sups[k] = sups[k] + g.sup
            else:
                infs[k] = infs[k] + float(np.min(g))
                sups[k] = sups[k] + float(np.max(g))
        
        # MATLAB: infs = fliplr(infs); sups = fliplr(sups);
        # fliplr flips left-right, equivalent to [::-1] in Python
        infs = infs[::-1]
        sups = sups[::-1]
        
        # Create polynomial functions (replacing poly2sym and matlabFunction)
        # Use numpy.poly1d for polynomial evaluation
        inf_poly = np.poly1d(infs)
        sup_poly = np.poly1d(sups)
        
        # Determine the left bound of new interval
        l = dom_index_inf
        
        inf_val_at_inf = inf_poly(dom_index_inf)
        sup_val_at_inf = sup_poly(dom_index_inf)
        
        if inf_val_at_inf <= 0 and sup_val_at_inf >= 0:
            l = dom_index_inf
        
        # inf(g[Y])(inf(x)) > 0
        if inf_val_at_inf > 0:
            # Select inf(g[Y]) for contraction
            if d < 4:
                # Use explicit analytical expressions
                l_roots = np.roots(infs)
                l_real = l_roots[np.isreal(l_roots)]
                if len(l_real) > 0:
                    l_real = np.real(l_real)
                    l_candidates = l_real[l_real > l]
                    if len(l_candidates) > 0:
                        l = np.min(l_candidates)
            else:
                # Use forward-backward contractor to find zero crossings
                # Create function handle for inf polynomial
                def infHan(x_val):
                    return inf_poly(x_val)
                
                try:
                    int_contract = contractForwardBackward(infHan, Interval(dom_index_inf, dom_index_sup))
                    if int_contract is not None and not int_contract.representsa_('emptySet', np.finfo(float).eps):
                        l = int_contract.inf if np.isscalar(int_contract.inf) else int_contract.inf[0]
                except Exception:
                    pass  # If contractForwardBackward fails, keep current l
        
        # sup(g[Y])(inf(x)) < 0
        if sup_val_at_inf < 0:
            # Select sup(g[Y]) for contraction
            if d < 4:
                sup_roots = np.roots(sups)
                sup_real = sup_roots[np.isreal(sup_roots)]
                if len(sup_real) > 0:
                    sup_real = np.real(sup_real)
                    sup_candidates = sup_real[sup_real > l]
                    if len(sup_candidates) > 0:
                        l = np.min(sup_candidates)
            else:
                # Use forward-backward contractor
                def supHan(x_val):
                    return sup_poly(x_val)
                
                try:
                    int_contract = contractForwardBackward(supHan, Interval(dom_index_inf, dom_index_sup))
                    if int_contract is not None and not int_contract.representsa_('emptySet', np.finfo(float).eps):
                        l = int_contract.inf if np.isscalar(int_contract.inf) else int_contract.inf[0]
                except Exception:
                    pass
        
        # Determine the right bound of new interval
        r = dom_index_sup
        
        inf_val_at_sup = inf_poly(dom_index_sup)
        sup_val_at_sup = sup_poly(dom_index_sup)
        
        if inf_val_at_sup <= 0 and sup_val_at_sup >= 0:
            r = dom_index_sup
        # inf(g[Y])(sup(x)) > 0
        elif inf_val_at_sup > 0:
            # Select inf(g[Y]) for contraction
            if d < 4:
                r_roots = np.roots(infs)
                r_real = r_roots[np.isreal(r_roots)]
                if len(r_real) > 0:
                    r_real = np.real(r_real)
                    r_candidates = r_real[r_real < r]
                    if len(r_candidates) > 0:
                        r = np.max(r_candidates)
            else:
                # Use forward-backward contractor
                def infHan(x_val):
                    return inf_poly(x_val)
                
                try:
                    int_contract = contractForwardBackward(infHan, Interval(dom_index_inf, dom_index_sup))
                    if int_contract is not None and not int_contract.representsa_('emptySet', np.finfo(float).eps):
                        r = int_contract.sup if np.isscalar(int_contract.sup) else int_contract.sup[0]
                except Exception:
                    pass
        # sup(g[Y])(sup(x)) < 0
        elif sup_val_at_sup < 0:
            # Select sup(g[Y]) for contraction
            if d < 4:
                r_roots = np.roots(sups)
                r_real = r_roots[np.isreal(r_roots)]
                if len(r_real) > 0:
                    r_real = np.real(r_real)
                    r_candidates = r_real[r_real < r]
                    if len(r_candidates) > 0:
                        r = np.max(r_candidates)
            else:
                # Use forward-backward contractor
                def supHan(x_val):
                    return sup_poly(x_val)
                
                try:
                    int_contract = contractForwardBackward(supHan, Interval(dom_index_inf, dom_index_sup))
                    if int_contract is not None and not int_contract.representsa_('emptySet', np.finfo(float).eps):
                        r = int_contract.sup if np.isscalar(int_contract.sup) else int_contract.sup[0]
                except Exception:
                    pass
        
        res_ = np.array([l, r])
        
    else:  # Independent variable
        # Get coefficient in front of the current variable
        ind = index - E.shape[0]
        if GI.size > 0 and ind < GI.shape[1]:
            coeff = GI[:, ind]
        else:
            coeff = np.array([0])
        
        # Check if domain can be updated
        coeff_abs = np.abs(coeff).max() if hasattr(coeff, 'max') and hasattr(coeff, 'ndim') and coeff.ndim > 0 else np.abs(coeff)
        if coeff_abs > np.finfo(float).eps:
            # Set generator for current variable to 0
            GI_new = GI.copy() if GI.size > 0 else np.array([])
            if GI_new.size > 0 and ind < GI_new.shape[1]:
                GI_new[:, ind] = 0
            
            # Compute bounds for function without current variable
            # c might be scalar or array
            c_arr = np.array([c]) if np.isscalar(c) else c
            g = -_aux_funcPoly(dom, c_arr, G, GI_new, E) / coeff
            
            # Extract new upper and lower bound from value
            if isinstance(g, Interval):
                res_ = np.array([g.inf if np.isscalar(g.inf) else g.inf[0], 
                                g.sup if np.isscalar(g.sup) else g.sup[0]])
            else:
                g_arr = np.asarray(g).flatten()
                res_ = np.array([np.min(g_arr), np.max(g_arr)])
        else:
            dom_index_inf = dom.inf[index] if dom.inf.size > index else dom.inf
            dom_index_sup = dom.sup[index] if dom.sup.size > index else dom.sup
            res_ = np.array([dom_index_inf, dom_index_sup])
    
    return res_

