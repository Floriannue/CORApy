"""
writeHigherOrderTensorFiles - create tensor files with order > 3

Syntax:
    writeHigherOrderTensorFiles(fdyn,vars,varsDer,path,fname,options)

Inputs:
    fdyn - symbolic function
    vars - struct containing the symbolic variables of the function
    varsDer - struct containing the symbolic derivatives of the variables
    path - file-path to the folder where the generated files are stored
    fname - name of the dynamical system
    options - struct containing the algorithm options

Outputs:
    -

Example:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Niklas Kochdumper
Written:       08-February-2018
Last update:   02-February-2021 (MW, remove code duplicates)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import os
import sympy as sp
import numpy as np
from typing import Any, Dict, List
from cora_python.g.functions.helper.dynamics.contDynamics.contDynamics.generateNthTensor import generateNthTensor
from cora_python.g.functions.helper.dynamics.contDynamics.contDynamics.evalNthTensor import evalNthTensor
from cora_python.g.functions.verbose.write.matlabFunction import matlabFunction


def writeHigherOrderTensorFiles(fdyn: Any, vars: Dict[str, Any], varsDer: Dict[str, Any],
                                path: str, fname: str, options: Dict[str, Any]) -> None:
    """
    Create tensor files with order > 3
    
    Args:
        fdyn: symbolic function
        vars: dict containing the symbolic variables of the function (keys: 'x', 'u', 'p')
        varsDer: dict containing the symbolic derivatives of the variables (keys: 'dx', 'du')
        path: file-path to the folder where the generated files are stored
        fname: name of the dynamical system
        options: dict containing the algorithm options (must contain 'tensorOrder')
    """
    
    # concatenate state and inputs
    # MATLAB: z = [vars.x;vars.u];
    if isinstance(vars['x'], (list, tuple)):
        z = list(vars['x']) + list(vars.get('u', []))
    else:
        z = [vars['x']] + ([vars['u']] if 'u' in vars and vars['u'] is not None else [])
    
    # MATLAB: dz = [varsDer.x;varsDer.u];
    if isinstance(varsDer.get('dx'), (list, tuple)):
        dz = list(varsDer.get('dx', [])) + list(varsDer.get('du', []))
    else:
        dz = ([varsDer.get('dx')] if varsDer.get('dx') is not None else []) + \
             ([varsDer.get('du')] if varsDer.get('du') is not None else [])
    
    # init for first call of generateNthTensor (order 4)
    # MATLAB: tensor = [];
    tensor = None
    
    # generate all higher-order tensors
    # MATLAB: for i = 4:options.tensorOrder
    for i in range(4, options['tensorOrder'] + 1):
        # MATLAB: tensor = generateNthTensor(fdyn,z,i,tensor);
        tensor = generateNthTensor(fdyn, z, i, tensor)
        # MATLAB: func = evalNthTensor(tensor,dz,i);
        func = evalNthTensor(tensor, dz, i)
        # MATLAB: func = aux_simplification(func,options,dz);
        func = aux_simplification(func, options, dz)
        # MATLAB: str = sprintf('tensor%i_%s',i,fname);
        str_name = f'tensor{i}_{fname}'
        # MATLAB: pathFile = [path, filesep, str];
        pathFile = os.path.join(path, str_name)
        # MATLAB: matlabFunction(func,'File',pathFile,'Vars',...
        #        {vars.x,vars.u,varsDer.x,varsDer.u,vars.p});
        var_list = []
        if vars.get('x') is not None:
            var_list.append(vars['x'])
        if vars.get('u') is not None:
            var_list.append(vars['u'])
        if varsDer.get('dx') is not None:
            var_list.append(varsDer['dx'])
        if varsDer.get('du') is not None:
            var_list.append(varsDer['du'])
        if vars.get('p') is not None:
            var_list.append(vars['p'])
        
        matlabFunction(func, File=pathFile, Vars=var_list)
        # MATLAB: if options.verbose
        if options.get('verbose', False):
            # MATLAB: disp(['... compute symbolic tensor ' num2str(i) 'th-order']);
            print(f'... compute symbolic tensor {i}th-order')


# Auxiliary functions -----------------------------------------------------

def aux_simplification(func: Any, options: Dict[str, Any], dz: Any) -> Any:
    """
    Simplifies the symbolic expression "func" with the specified method
    
    Args:
        func: symbolic expression to simplify
        options: dict containing algorithm options
        dz: symbolic derivative variables
        
    Returns:
        func: simplified expression
    """
    
    # MATLAB: if isfield(options,'lagrangeRem')
    if 'lagrangeRem' in options:
        # MATLAB: if isfield(options.lagrangeRem,'simplify')
        if 'simplify' in options['lagrangeRem']:
            # MATLAB: if strcmp(options.lagrangeRem.simplify,'simplify')
            if options['lagrangeRem']['simplify'] == 'simplify':
                # MATLAB: func = simplify(func);
                func = sp.simplify(func)
            # MATLAB: elseif strcmp(options.lagrangeRem.simplify,'collect')
            elif options['lagrangeRem']['simplify'] == 'collect':
                # MATLAB: func = collect(func,dz);
                func = sp.collect(func, dz)
    
    return func

