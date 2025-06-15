"""
CORAlinprog - evaluates a linear program for MATLAB and MOSEK syntax;
   we need this function for the following reasons: MATLAB's linprog
   function could previously be called by
      linprog(f,A,b,Ae,be,lb,ub,x0,options)
   which was recently changed to
      linprog(f,A,b,Ae,be,lb,ub,options)
   forcing us to use the alternative syntax
      linprog(problem)
   where problem is a struct. However, the MOSEK overload of linprog
   cannot handle that syntax, so we use this wrapper, instead.

   Since the dual-simplex algorithm sometimes returns exitflag = -9 for
   problems which do have a solution, we have an automatic switch to the
   interior-point solver in that case. Note that we use the dual-simplex
   algorithm by default since it has shown to be more accurate.

Syntax:
    x, fval, exitflag, output, lambda = CORAlinprog(problem)

Inputs:
    problem - linear program definition, with fields
              - problem.f (cost function min f*x)
              - problem.Aineq (inequality constraint Aineq * x <= bineq)
              - problem.bineq (inequality constraint Aineq * x <= bineq)
              - problem.Aeq (equality constraint Aeq * x == beq)
              - problem.beq (equality constraint Aeq * x == beq)
              - problem.lb (lower bound for optimization variable)
              - problem.ub (upper bound for optimization variable)
              - problem.x0 (initial point)
              where all numeric values should be of type double.

Outputs:
    x - minimizer
    fval - minimal objective value
    exitflag - status of linear program
    output - further output of scipy linprog
    lambda - further output of scipy linprog

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger
         Python translation by AI Assistant
Written: 16-July-2024
Last update: 04-October-2024 (MW, switch to interior-point in MATLAB call)
             09-October-2024 (TL, compatibility >=R2024a)
             29-October-2024 (TL, problem fields should be doubles)
Last revision: ---
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, Union
from scipy.optimize import linprog
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
import warnings


def CORAlinprog(problem: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[float], int, Dict, Dict]:
    """
    Evaluates a linear program using scipy.optimize.linprog.
    
    Args:
        problem: Dictionary containing linear program definition with fields:
                - f: cost function coefficients (min f*x)
                - Aineq: inequality constraint matrix (Aineq * x <= bineq)
                - bineq: inequality constraint vector
                - Aeq: equality constraint matrix (Aeq * x == beq)
                - beq: equality constraint vector
                - lb: lower bounds for optimization variable
                - ub: upper bounds for optimization variable
                - x0: initial point (not used in scipy)
                - options: solver options (optional)
                
    Returns:
        x: minimizer (None if infeasible)
        fval: minimal objective value (None if infeasible)
        exitflag: status of linear program (1=optimal, -2=infeasible, -3=unbounded, 0=other)
        output: further output information
        lambda: dual variables (empty dict in this implementation)
    """
    # Extract problem components
    c = problem.get('f', None)
    A_ub = problem.get('Aineq', None)
    b_ub = problem.get('bineq', None)
    A_eq = problem.get('Aeq', None)
    b_eq = problem.get('beq', None)
    bounds = None
    
    # Handle bounds
    lb = problem.get('lb', None)
    ub = problem.get('ub', None)
    
    if lb is not None or ub is not None:
        n_vars = len(c) if c is not None else 0
        if n_vars == 0 and A_ub is not None:
            n_vars = A_ub.shape[1]
        elif n_vars == 0 and A_eq is not None:
            n_vars = A_eq.shape[1]
            
        if n_vars > 0:
            bounds = []
            for i in range(n_vars):
                lower = lb[i] if lb is not None and i < len(lb) else None
                upper = ub[i] if ub is not None and i < len(ub) else None
                bounds.append((lower, upper))
    
    # Convert to numpy arrays if needed
    if c is not None:
        c = np.array(c, dtype=float).flatten()
    if A_ub is not None:
        A_ub = np.array(A_ub, dtype=float)
    if b_ub is not None:
        b_ub = np.array(b_ub, dtype=float).flatten()
    if A_eq is not None:
        A_eq = np.array(A_eq, dtype=float)
    if b_eq is not None:
        b_eq = np.array(b_eq, dtype=float).flatten()
    
    # Set solver options
    method = 'highs'  # Default to HiGHS solver (most robust)
    options_dict = {}
    
    if 'options' in problem and problem['options'] is not None:
        # Try to extract relevant options
        opts = problem['options']
        if hasattr(opts, 'Algorithm'):
            if 'dual-simplex' in opts.Algorithm.lower():
                method = 'highs-ds'
            elif 'interior-point' in opts.Algorithm.lower():
                method = 'highs-ipm'
        if hasattr(opts, 'Display') and opts.Display.lower() == 'off':
            options_dict['disp'] = False
    
    # Try multiple methods if first one fails
    methods_to_try = [method, 'highs', 'highs-ds', 'highs-ipm']
    methods_to_try = list(dict.fromkeys(methods_to_try))  # Remove duplicates while preserving order
    
    result = None
    for method_attempt in methods_to_try:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = linprog(
                    c=c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=bounds,
                    method=method_attempt,
                    options=options_dict
                )
            
            # If successful, break
            if result.success or result.status in [0, 1]:
                break
                
        except Exception as e:
            # Continue to next method
            continue
    
    # Process results
    if result is None:
        # All methods failed
        x = None
        fval = None
        exitflag = 0
        output = {'message': 'All solver methods failed'}
        lambda_out = {}
    else:
        # Map scipy results to MATLAB-like outputs
        if result.success:
            x = result.x.reshape(-1, 1) if result.x is not None else None
            fval = result.fun
            exitflag = 1
        elif result.status == 2:  # Infeasible
            x = None
            fval = None
            exitflag = -2
        elif result.status == 3:  # Unbounded
            x = None
            fval = None
            exitflag = -3
        else:
            x = None
            fval = None
            exitflag = 0
        
        # Create output structure
        output = {
            'message': result.message,
            'iterations': getattr(result, 'nit', 0),
            'algorithm': method,
            'status': result.status
        }
        
        # Lambda (dual variables) - scipy provides these differently
        lambda_out = {}
        if hasattr(result, 'ineqlin') and result.ineqlin is not None:
            lambda_out['ineqlin'] = result.ineqlin
        if hasattr(result, 'eqlin') and result.eqlin is not None:
            lambda_out['eqlin'] = result.eqlin
        if hasattr(result, 'lower') and result.lower is not None:
            lambda_out['lower'] = result.lower
        if hasattr(result, 'upper') and result.upper is not None:
            lambda_out['upper'] = result.upper
    
    return x, fval, exitflag, output, lambda_out 