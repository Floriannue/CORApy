"""
derivatives - compute derivatives of nonlinear reset functions

Syntax:
    derivatives(nonlinReset)
    nonlinReset = derivatives(nonlinReset)
    nonlinReset = derivatives(nonlinReset,fpath)
    nonlinReset = derivatives(nonlinReset,fpath,fname)
    nonlinReset = derivatives(nonlinReset,fpath,fname,tensorOrder)

Inputs:
    nonlinReset - nonlinearReset object
    fpath - where to store generated files
    fname - name of reset function, used for file names
    tensorOrder - tensor order (1, 2, or 3)

Outputs:
    nonlinReset - nonlinearReset object with set properties for derivatives

Example:
    nonlinReset = nonlinearReset(lambda x, u: np.array([[x[0]*x[1]], [x[1]]]))
    path = os.path.join(CORAROOT(), 'models', 'auxiliary')
    fname = 'example_derivatives'
    nonlinReset = derivatives(nonlinReset, path, fname, 2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       12-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import os
import sys
import sympy as sp
import numpy as np
from typing import Any, Optional
from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.g.functions.verbose.write.derive import derive


def derivatives(nonlinReset: NonlinearReset, *varargin) -> NonlinearReset:
    """
    Compute derivatives of nonlinear reset functions
    
    Args:
        nonlinReset: nonlinearReset object
        *varargin: optional arguments:
            fpath: str - where to store generated files
            fname: str - name of reset function, used for file names
            tensorOrder: int - tensor order (1, 2, or 3)
            
    Returns:
        nonlinReset: nonlinearReset object with set properties for derivatives
    """
    
    # parse input arguments
    # MATLAB: narginchk(1,5);
    if len(varargin) > 4:
        raise TypeError("derivatives expects at most 4 arguments")
    
    defaultPath = os.path.join(CORAROOT(), 'models', 'auxiliary', 'nonlinearReset')
    fpath, fname, tensorOrder = setDefaultValues([defaultPath, 'nonlinearReset', 2], list(varargin))
    
    # in derivatives computation
    if not os.path.exists(fpath):
        os.makedirs(fpath, exist_ok=True)
    
    # Add path to sys.path so generated files can be imported
    if fpath not in sys.path:
        sys.path.insert(0, fpath)
    
    # note: currently, we override the derivatives at new call of 'derivatives'
    # if the file path is the same
    
    # save tensor order
    nonlinReset.tensorOrder = tensorOrder
    
    # generate symbolic variables
    # MATLAB: x = sym('x',[nonlinReset.preStateDim,1],'real');
    x = [sp.Symbol(f'x{i+1}', real=True) for i in range(nonlinReset.preStateDim)]
    # MATLAB: u = sym('u',[nonlinReset.inputDim,1],'real');
    u = [sp.Symbol(f'u{i+1}', real=True) for i in range(nonlinReset.inputDim)]
    vars = [x, u]
    varNamesIn = ['x', 'u']
    
    # Jacobian file
    # MATLAB: [J_cell,J_han] = derive('FunctionHandle',nonlinReset.f,...)
    J_cell, J_han = derive('FunctionHandle', nonlinReset.f,
                          'Vars', vars, 'VarNamesIn', varNamesIn, 'VarNamesOut', ['A', 'B'],
                          'Path', fpath, 'FileName', f'{fname}_jacobian')
    nonlinReset.J = J_han
    # note: when executing J_han(vars{:}), one obtains
    #   J_cell[0] = out1, J_cell[1] = out2, ...
    
    if tensorOrder == 1:
        return nonlinReset
    
    # rewrite for further derivation
    # MATLAB: J = [J_cell{1}, J_cell{2}];
    # J_cell[0]: df1/dx1,dx2,... df2/dx1,dx2,...
    # J_cell[1]: df1/du1,du2,... df2/du1,du2,...
    if isinstance(J_cell[0], sp.Matrix) and isinstance(J_cell[1], sp.Matrix):
        J = J_cell[0].row_join(J_cell[1])
    else:
        # Convert to matrices if needed
        J0 = sp.Matrix(J_cell[0]) if not isinstance(J_cell[0], sp.Matrix) else J_cell[0]
        J1 = sp.Matrix(J_cell[1]) if not isinstance(J_cell[1], sp.Matrix) else J_cell[1]
        J = J0.row_join(J1)
    
    # Hessian file
    # MATLAB: [H_cell,H_han] = derive('SymbolicFunction',J,...)
    H_cell, H_han = derive('SymbolicFunction', J,
                           'Vars', vars, 'VarNamesIn', varNamesIn, 'VarNamesOut', ['Hx', 'Hu'],
                           'Path', fpath, 'FileName', f'{fname}_hessian',
                           'IntervalArithmetic', tensorOrder == 2)
    nonlinReset.H = H_han
    # note: when executing H_han(vars{:}), one obtains
    #   H_cell[0] = out1, H_cell[1] = out2, ...
    
    if tensorOrder == 2:
        return nonlinReset
    
    # rewrite for further derivation
    # MATLAB: H = [H_cell{1}, H_cell{2}];
    if isinstance(H_cell[0], sp.Matrix) and isinstance(H_cell[1], sp.Matrix):
        H = H_cell[0].row_join(H_cell[1])
    else:
        # Convert to matrices if needed
        H0 = sp.Matrix(H_cell[0]) if not isinstance(H_cell[0], sp.Matrix) else H_cell[0]
        H1 = sp.Matrix(H_cell[1]) if not isinstance(H_cell[1], sp.Matrix) else H_cell[1]
        H = H0.row_join(H1)
    
    # third-order tensor file
    # MATLAB: [~,T_han] = derive('SymbolicFunction',H,...)
    _, T_han = derive('SymbolicFunction', H,
                     'Vars', vars, 'VarNamesIn', varNamesIn, 'VarNamesOut', ['Tx', 'Tu'],
                     'Path', fpath, 'FileName', f'{fname}_thirdOrderTensor',
                     'IntervalArithmetic', tensorOrder == 3)
    nonlinReset.T = T_han
    
    return nonlinReset

