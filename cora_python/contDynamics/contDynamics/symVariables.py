"""
symVariables - generates symbolic variables of a continuous system; the
   symbolic variables are either set or a 0x1 sym object

Syntax:
    [vars,vars_der] = symVariables(sys)
    [vars,vars_der] = symVariables(sys,withBrackets)

Inputs:
    sys - contDynamics object
    withBrackets - true/false
       true:  variable 'x' results in symbolic variables x1, x2, ...
       false: variable 'x' results in symbolic variables xL1R, xL2R, ...

Outputs:
    vars - struct with fields
       .x - symbolic state variables
       .u - symbolic input variables
       .y - symbolic constraint variables
       .o - symbolic output variables
       .p - symbolic parameters
    vars_der - struct with fields
       .dx - symbolic state deviation from linearization point
       .du - symbolic input deviation from linearization point
       .dy - symbolic constraint deviation from linearization point
       .do - symbolic output deviation from linearization point

Example: 
    sys = contDynamics('test',3,1,2);
    [vars,vars_der] = symVariables(sys);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: derivatives

Authors:       Matthias Althoff, Mark Wetzlinger, Tobias Ladner
Written:       18-January-2008
Last update:   06-July-2017
               05-November-2017
               14-January-2018
               19-November-2022 (MW, add outputs)
               03-February-2023 (SM, real symbolic variables)
               05-March-2024 (LL, consider nonlinearARX sys)
               07-October-2024 (MW, use auxiliary function)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import sympy as sp
import numpy as np
from typing import Tuple, Dict, Any, Optional
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.string.bracketSubs import bracketSubs


def symVariables(sys: Any, *varargin) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generates symbolic variables of a continuous system
    
    Args:
        sys: contDynamics object
        *varargin: optional withBrackets (bool)
            True:  variable 'x' results in symbolic variables x1, x2, ...
            False: variable 'x' results in symbolic variables xL1R, xL2R, ...
        
    Returns:
        vars: dict with fields
            'x': symbolic state variables
            'u': symbolic input variables
            'y': symbolic constraint variables
            'o': symbolic output variables
            'p': symbolic parameters
        vars_der: dict with fields
            'dx': symbolic state deviation from linearization point
            'du': symbolic input deviation from linearization point
            'dy': symbolic constraint deviation from linearization point
            'do': symbolic output deviation from linearization point
    """
    
    # MATLAB: narginchk(1,2);
    # MATLAB: withBrackets = setDefaultValues({false},varargin);
    withBrackets = setDefaultValues([False], list(varargin))[0]
    
    # TODO: fix possibly wrong usage of sys.nrOfDims for nonlinearARX and
    #       harmonize calls below
    
    # symbolic variables
    # MATLAB: vars = struct('x',[],'u',[],'y',[],'o',[],'p',[]);
    vars_dict = {'x': None, 'u': None, 'y': None, 'o': None, 'p': None}
    # symbolic variables for deviation from linearization point
    # MATLAB: vars_der = struct('x',[],'u',[],'y',[],'o',[]);
    vars_der = {'dx': None, 'du': None, 'dy': None, 'do': None}
    
    # states and inputs
    # MATLAB: if isa(sys,'nonlinearARX')
    is_nonlinearARX = hasattr(sys, '__class__') and 'nonlinearARX' in sys.__class__.__name__.lower()
    
    if is_nonlinearARX:
        # MATLAB: vars.x = aux_symVector('x',sys.n_p*sys.nrOfOutputs,withBrackets);
        # Use Python naming convention first
        n_p = getattr(sys, 'n_p', 0)
        
        nr_of_outputs = getattr(sys, 'nr_of_outputs', None)
        if nr_of_outputs is None:
            nr_of_outputs = getattr(sys, 'nrOfOutputs', 0)
        
        nr_of_inputs = getattr(sys, 'nr_of_inputs', None)
        if nr_of_inputs is None:
            nr_of_inputs = getattr(sys, 'nrOfInputs', 0)
        
        vars_dict['x'] = aux_symVector('x', n_p * nr_of_outputs, withBrackets)
        # MATLAB: vars_der.x = aux_symVector('dx',sys.n_p*sys.nrOfOutputs,withBrackets);
        vars_der['dx'] = aux_symVector('dx', n_p * nr_of_outputs, withBrackets)
        # MATLAB: vars.u = aux_symVector('u',(sys.n_p+1)*sys.nrOfInputs,withBrackets);
        vars_dict['u'] = aux_symVector('u', (n_p + 1) * nr_of_inputs, withBrackets)
        # MATLAB: vars_der.u = aux_symVector('du',(sys.n_p+1)*sys.nrOfInputs,withBrackets);
        vars_der['du'] = aux_symVector('du', (n_p + 1) * nr_of_inputs, withBrackets)
    else:
        # MATLAB: vars.x = aux_symVector('x',sys.nrOfDims,withBrackets);
        # Use Python naming convention first, fallback to MATLAB
        # For required properties, 0 is a reasonable default (empty system is valid)
        nr_of_dims = getattr(sys, 'nr_of_dims', None)
        if nr_of_dims is None:
            nr_of_dims = getattr(sys, 'nrOfDims', 0)
        
        nr_of_inputs = getattr(sys, 'nr_of_inputs', None)
        if nr_of_inputs is None:
            nr_of_inputs = getattr(sys, 'nrOfInputs', 0)
        
        vars_dict['x'] = aux_symVector('x', nr_of_dims, withBrackets)
        # MATLAB: vars_der.x = aux_symVector('dx',sys.nrOfDims,withBrackets);
        vars_der['dx'] = aux_symVector('dx', nr_of_dims, withBrackets)
        # MATLAB: vars.u = aux_symVector('u',sys.nrOfInputs,withBrackets);
        vars_dict['u'] = aux_symVector('u', nr_of_inputs, withBrackets)
        # MATLAB: vars_der.u = aux_symVector('du',sys.nrOfInputs,withBrackets);
        vars_der['du'] = aux_symVector('du', nr_of_inputs, withBrackets)
    
    # algebraic constraints
    # MATLAB: if isprop(sys,'nrOfConstraints')
    # Try Python naming first, then MATLAB naming
    nr_of_constraints = getattr(sys, 'nr_of_constraints', None)
    if nr_of_constraints is None:
        nr_of_constraints = getattr(sys, 'nrOfConstraints', None)
    
    if nr_of_constraints is not None:
        # MATLAB: vars.y = aux_symVector('y',sys.nrOfConstraints,withBrackets);
        vars_dict['y'] = aux_symVector('y', nr_of_constraints, withBrackets)
        # MATLAB: vars_der.y = aux_symVector('dy',sys.nrOfConstraints,withBrackets);
        vars_der['dy'] = aux_symVector('dy', nr_of_constraints, withBrackets)
    else:
        # MATLAB: vars.y = sym('y',[0,1]);
        vars_dict['y'] = sp.Matrix([])  # Empty symbolic matrix
        # MATLAB: vars_der.y = sym('dy',[0,1]);
        vars_der['dy'] = sp.Matrix([])  # Empty symbolic matrix
    
    # outputs
    # MATLAB: vars.o = aux_symVector('y',sys.nrOfOutputs,withBrackets);
    # Use Python naming convention first, fallback to MATLAB
    # For required properties, 0 is a reasonable default (empty system)
    nr_of_outputs = getattr(sys, 'nr_of_outputs', None)
    if nr_of_outputs is None:
        nr_of_outputs = getattr(sys, 'nrOfOutputs', 0)
    
    vars_dict['o'] = aux_symVector('y', nr_of_outputs, withBrackets)
    # MATLAB: vars_der.o = aux_symVector('do',sys.nrOfOutputs,withBrackets);
    vars_der['do'] = aux_symVector('do', nr_of_outputs, withBrackets)
    
    # parameters
    # MATLAB: if isprop(sys,'nrOfParam')
    # Try Python naming first, then MATLAB naming
    nr_of_param = getattr(sys, 'nr_of_param', None)
    if nr_of_param is None:
        nr_of_param = getattr(sys, 'nrOfParam', None)
    
    if nr_of_param is not None:
        # MATLAB: vars.p = aux_symVector('p',sys.nrOfParam,withBrackets);
        vars_dict['p'] = aux_symVector('p', nr_of_param, withBrackets)
    else:
        # MATLAB: vars.p = sym('p',[0,1]);
        vars_dict['p'] = sp.Matrix([])  # Empty symbolic matrix
    
    return vars_dict, vars_der


# Auxiliary functions -----------------------------------------------------

def aux_symVector(varName: str, nrVars: int, withBrackets: bool) -> Any:
    """
    Creates a symbolic vector with the given variable name and number of variables
    
    withBrackets == true sandwiches the number in between 'L' and 'R'
    
    Args:
        varName: base name for the symbolic variables
        nrVars: number of variables to create
        withBrackets: if True, use format 'xL1R', 'xL2R', etc.; if False, use 'x1', 'x2', etc.
        
    Returns:
        symVars: symbolic vector (sympy Matrix or single symbol)
    """
    
    if withBrackets:
        # apparently, this case differentiation below is necessary...
        # MATLAB: if nrVars == 1
        if nrVars == 1:
            # MATLAB: symVars = sym([varName 'L%dR'],[2,1],'real');
            # MATLAB: symVars = symVars(1);
            # Create symbol with bracket format, then extract first element
            # In sympy, we create symbols with the format 'xL1R', 'xL2R', etc.
            # But for nrVars==1, MATLAB creates [2,1] then takes first, so we create just one
            sym_str = f'{varName}L1R'
            symVars = sp.Symbol(sym_str, real=True)
        else:
            # MATLAB: symVars = sym([varName 'L%dR'],[nrVars,1],'real');
            # Create symbols with bracket format: 'xL1R', 'xL2R', ..., 'xLnR'
            sym_strs = [f'{varName}L{i+1}R' for i in range(nrVars)]
            symVars = sp.Matrix([sp.Symbol(s, real=True) for s in sym_strs])
    else:
        # MATLAB: symVars = sym(varName,[nrVars,1],'real');
        # Create symbols with simple format: 'x1', 'x2', ..., 'xn'
        if nrVars == 1:
            symVars = sp.Symbol(f'{varName}1', real=True)
        else:
            sym_strs = [f'{varName}{i+1}' for i in range(nrVars)]
            symVars = sp.Matrix([sp.Symbol(s, real=True) for s in sym_strs])
    
    return symVars

