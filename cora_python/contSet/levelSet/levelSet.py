"""
Level set class

A level set is defined as:
L := {x ∈ R^n | eq(x) ∼ 0}
where ∼ ∈ {==, <=, <}

Properties:
    eq: symbolic equation
    vars: symbolic variables
    compOp: comparison operator ('==', '<=', '<')
    solved: solved equations (optional)

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 19-July-2017 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, Any, Tuple, List, Dict, Callable
from ..contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.macros import CHECKS_ENABLED

# Import symbolic computation capabilities
try:
    import sympy as sp
    from sympy import symbols, diff, hessian, solve, lambdify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

class LevelSet(ContSet):
    """
    Level set class
    
    A level set represents sets of the form:
    L := {x ∈ R^n | eq(x) ∼ 0} where ∼ ∈ {==, <=, <}
    """
    
    def __init__(self, *varargin):
        """
        Constructor for level set objects
        
        Args:
            *varargin: Variable arguments
                     - levelSet(eq, vars, compOp, [solved])
                     - levelSet(other_levelSet): copy constructor
        """
        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        
        # Check number of input arguments
        if len(varargin) < 1 or len(varargin) > 4:
            raise CORAerror('CORA:wrongInputInConstructor', f'Expected 1-4 arguments, got {len(varargin)}')

        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], LevelSet):
            other = varargin[0]
            self.eq = other.eq
            self.vars = other.vars
            self.compOp = other.compOp
            self.solved = other.solved if hasattr(other, 'solved') else None
            self.funHan = other.funHan if hasattr(other, 'funHan') else None
            self.der = other.der if hasattr(other, 'der') else None
            self._dim_val = other._dim_val if hasattr(other, '_dim_val') else (other.dim() if hasattr(other, 'dim') and callable(other.dim) else 0)
            self.solvable = other.solvable if hasattr(other, 'solvable') else False
            super().__init__()
            self.precedence = 20
            return

        # 2. parse input arguments: varargin -> vars
        eq, vars_, compOp, solved = _aux_parseInputArgs(*varargin)

        # 3. check correctness of input arguments
        _aux_checkInputArgs(eq, vars_, compOp, solved, len(varargin))

        # 4. compute internal properties
        eq, vars_, compOp, solved, funHan, der, dim, solvable = _aux_computeProperties(
            eq, vars_, compOp, solved, len(varargin))

        # 5. assign properties
        self.eq = eq
        self.vars = vars_
        self.compOp = compOp
        self.solved = solved
        self.funHan = funHan
        self.der = der
        self._dim_val = dim  # Store as private property, access via dim() method
        self.solvable = solvable

        # 6. set precedence (fixed) and initialize parent
        super().__init__()
        self.precedence = 20

    def __repr__(self):
        """String representation"""
        if hasattr(self, '_dim_val'):
            return f"LevelSet(dim={self._dim_val}, compOp='{self.compOp}')"
        else:
            return "LevelSet(empty)"


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin) -> Tuple[Any, Any, str, Optional[Any]]:
    """Parse input arguments from user and assign to variables"""
    
    # no input arguments
    if len(varargin) == 0:
        return None, None, '', None
    
    # set basic class properties
    eq = varargin[0]
    vars_ = varargin[1] if len(varargin) > 1 else []
    compOp = varargin[2] if len(varargin) > 2 else '=='
    solved = varargin[3] if len(varargin) > 3 else None
    
    return eq, vars_, compOp, solved


def _aux_checkInputArgs(eq: Any, vars_: Any, compOp: Union[str, List[str]], solved: Any, n_in: int):
    """Check correctness of input arguments"""
    
    # only check if macro set to true
    if CHECKS_ENABLED and n_in > 0:
        
        # Check for length mismatch between equations and operators
        if isinstance(eq, list) and isinstance(compOp, str):
            raise CORAerror('CORA:wrongValue', 'third', "must be a list of comparison operators.")
        
        if isinstance(eq, list) and isinstance(compOp, list) and len(eq) != len(compOp):
            raise CORAerror('CORA:dimensionMismatch', 'eq and compOp must have the same length.')

        # check comparison operator values
        if not isinstance(compOp, list):
            # single comparison operator has to be '==', '<=', '<'
            if compOp not in ['==', '<=', '<']:
                raise CORAerror('CORA:wrongValue', 'third', "be '==' or '<=' or '<'")
        else:
            # for multiple comparison operators, no '==' allowed
            for op in compOp:
                if op not in ['<=', '<']:
                    raise CORAerror('CORA:wrongValue', 'third', "be '<=' or '<'")


def _aux_computeProperties(eq: Any, vars_: Any, compOp: str, solved: Any, n_in: int) -> Tuple[Any, Any, str, Any, Any, Any, int, bool]:
    """Compute properties according to given user inputs"""
    
    # set default values
    der = None
    solvable = False
    funHan = None
    
    # dimension
    dim = len(vars_) if vars_ else 0
    
    if n_in == 0:
        return eq, vars_, compOp, solved, funHan, der, dim, solvable
    
    if not SYMPY_AVAILABLE:
        # Fallback without symbolic computation
        return eq, vars_, compOp, solved, funHan, der, dim, solvable
    
    # Convert to SymPy if needed
    if not isinstance(eq, sp.Basic):
        # If eq is a string, parse it as SymPy expression
        if isinstance(eq, str):
            eq = sp.sympify(eq)
        else:
            # Try to convert other types
            try:
                eq = sp.sympify(eq)
            except:
                # Keep original if conversion fails
                pass
    
    # Convert vars to SymPy symbols if needed
    if isinstance(vars_, list) and len(vars_) > 0:
        if not isinstance(vars_[0], sp.Symbol):
            # Create SymPy symbols
            symbol_names = [str(var) for var in vars_]
            vars_ = [sp.Symbol(name) for name in symbol_names]
    
    # function handle (lambdify converts SymPy to callable function)
    if isinstance(eq, sp.Basic) and len(vars_) > 0:
        try:
            funHan = sp.lambdify(vars_, eq, 'numpy')
        except:
            funHan = None
    
    # compute derivatives for equality constraints
    if compOp == '==' and isinstance(eq, sp.Basic) and len(vars_) > 0:
        der = {}
        try:
            # gradient
            grad = [sp.diff(eq, var) for var in vars_]
            der['grad'] = sp.lambdify(vars_, grad, 'numpy')
            
            # hessian matrix
            hess = sp.hessian(eq, vars_)
            der['hess'] = sp.lambdify(vars_, hess, 'numpy')
            
            # third-order derivatives (tensor)
            third = []
            for grad_elem in grad:
                third_elem = [sp.diff(grad_elem, var) for var in vars_]
                third.append(sp.lambdify(vars_, third_elem, 'numpy'))
            der['third'] = third
            
        except Exception as e:
            # If derivative computation fails, keep partial results
            pass
    
    # try to solve non-linear equation for one variable
    if compOp == '==' and isinstance(eq, sp.Basic) and len(vars_) > 0:
        
        if solved is None:
            solved = []
            for i, var in enumerate(vars_):
                solved_info = {}
                
                # check if variable is contained in equation
                eq_vars = eq.free_symbols
                if var in eq_vars:
                    solved_info['contained'] = True
                    
                    try:
                        # solve equation for this variable
                        solutions = sp.solve(eq, var)
                        
                        if solutions:
                            solved_info['solvable'] = True
                            solved_info['eq'] = solutions
                            
                            # create function handles for each solution
                            solved_info['funHan'] = []
                            other_vars = [v for v in vars_ if v != var]
                            
                            for sol in solutions:
                                sol_info = {}
                                try:
                                    # equation
                                    sol_info['eq'] = sp.lambdify(other_vars, sol, 'numpy')
                                    
                                    # gradient (with respect to other variables)
                                    if len(other_vars) > 0:
                                        grad = [sp.diff(sol, v) for v in other_vars]
                                        sol_info['grad'] = sp.lambdify(other_vars, grad, 'numpy')
                                        
                                        # hessian
                                        hess = sp.hessian(sol, other_vars)
                                        sol_info['hess'] = sp.lambdify(other_vars, hess, 'numpy')
                                        
                                        # third-order
                                        third = []
                                        for grad_elem in grad:
                                            third_elem = [sp.diff(grad_elem, v) for v in other_vars]
                                            third.append(sp.lambdify(other_vars, third_elem, 'numpy'))
                                        sol_info['third'] = third
                                    
                                    solved_info['funHan'].append(sol_info)
                                except:
                                    # Skip this solution if function handle creation fails
                                    pass
                            
                            solvable = True
                            
                        else:
                            solved_info['solvable'] = False
                            solved_info['eq'] = []
                            solved_info['funHan'] = []
                            
                    except Exception as e:
                        solved_info['solvable'] = False
                        solved_info['eq'] = []
                        solved_info['funHan'] = []
                        
                else:
                    solved_info['contained'] = False
                    solved_info['solvable'] = False
                    solved_info['eq'] = []
                    solved_info['funHan'] = []
                
                solved.append(solved_info)
    
    return eq, vars_, compOp, solved, funHan, der, dim, solvable 