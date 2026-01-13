"""
derivatives - computes multivariate derivatives (Jacobians, Hessians, 
   third-order tensors, higher-order tensors) of nonlinear systems
   symbolically; the result is stored in Python files to which the
   contDynamics object has access via handles in its properties

Syntax:
    derivatives(sys)
    derivatives(sys,options)
    derivatives(sys,options,path)

Inputs:
    sys - contDynamics object
    options - options dict, with fields
       .tensorOrder
       .tensorOrderOutput
       .lagrangeRem.simplify
       .lagrangeRem.tensorParallel
       .lagrangeRem.method
       .lagrangeRem.replacements
       .verbose: output text on command window
    path - filepath to save generated files

Outputs:
    -

Example:
    f = lambda x, u: [x[0] - u[0], x[0]*x[1]]
    sys = nonlinearSys('nln',f)
    derivatives(sys)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
Written:       29-October-2007 (MA)
Last update:   07-September-2012 (MA)
               12-October-2015 (MA)
               08-April-2016 (MA)
               10-June-2017 (NK)
               15-June-2017 (NK)
               28-June-2017 (NK)
               06-July-2017
               15-July-2017 (NK)
               16-July-2017 (NK)
               05-November-2017 (MA, generalization for all contDynamics classes)
               12-November-2017 (MA)
               03-December-2017 (MA)
               14-January-2018 (MA)
               12-November-2018 (NK, removed lagrange remainder files)
               29-January-2021 (MW, simplify checks, outsource tensor check)
               18-November-2022 (MW, integrate output equation)
               22-March-2024 (LL, add verbose option)
Last revision: 07-October-2024 (MW, refactor)
               Automatic python translation: Florian Nüssel BA 2025
"""

import os
import sys as sys_module
import pickle
import sympy as sp
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.g.functions.helper.dynamics.checkOptions.getDefaultValue import getDefaultValue
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.init.unitvector import unitvector
from cora_python.contDynamics.contDynamics.symVariables import symVariables
from cora_python.contDynamics.contDynamics.private.priv_checkTensorRecomputation import priv_checkTensorRecomputation
from cora_python.g.functions.verbose.write.writeMatrixFile import writeMatrixFile
from cora_python.g.functions.verbose.write.writeHessianTensorFile import writeHessianTensorFile
from cora_python.g.functions.verbose.write.write3rdOrderTensorFile import write3rdOrderTensorFile
from cora_python.g.functions.verbose.write.writeHigherOrderTensorFiles import writeHigherOrderTensorFiles
from cora_python.contSet.interval import Interval


def derivatives(sys: Any, *varargin) -> None:
    """
    Computes multivariate derivatives (Jacobians, Hessians, third-order tensors,
    higher-order tensors) of nonlinear systems symbolically
    
    Args:
        sys: contDynamics object
        *varargin: optional arguments:
            options: dict with algorithm options
            path: str - filepath to save generated files
    """
    
    # MATLAB: narginchk(1,3);
    if len(varargin) > 2:
        raise TypeError("derivatives expects at most 3 arguments")
    
    # MATLAB: defaultOptions = aux_defaultOptions(sys);
    defaultOptions = aux_defaultOptions(sys)
    # MATLAB: defaultPath = [CORAROOT filesep 'models' filesep 'auxiliary' filesep sys.name];
    defaultPath = os.path.join(CORAROOT(), 'models', 'auxiliary', sys.name)
    # MATLAB: [options,path] = setDefaultValues({defaultOptions,defaultPath},varargin);
    options, path = setDefaultValues([defaultOptions, defaultPath], list(varargin))
    
    # be lenient with verbose output...
    # MATLAB: options.verbose = isfield(options,'verbose') && options.verbose;
    options['verbose'] = options.get('verbose', False) and options.get('verbose', False)
    
    # create symbolic variables
    # MATLAB: [vars,varsDer] = symVariables(sys,true);
    vars, varsDer = symVariables(sys, True)
    
    # insert symbolic variables into the system equations
    # MATLAB: [fdyn,fcon,fout] = aux_insertSymVariables(sys,vars);
    fdyn, fcon, fout = aux_insertSymVariables(sys, vars)
    
    # check if old derivatives can be re-used
    # MATLAB: [requiredFiles,requiredFiles_out,storedData,deleteAll] = ...
    #    priv_checkTensorRecomputation(sys,fdyn,fcon,fout,path,options);
    requiredFiles, requiredFiles_out, storedData, deleteAll = \
        priv_checkTensorRecomputation(sys, fdyn, fcon, fout, path, options)
    
    # already finished if nothing new required
    # MATLAB: if ~any([requiredFiles.standard;requiredFiles.int;requiredFiles.higherOrders;...])
    if not (np.any(requiredFiles['standard']) or np.any(requiredFiles['int']) or
            np.any(requiredFiles['higherOrders']) or
            np.any(requiredFiles_out['standard']) or np.any(requiredFiles_out['int']) or
            np.any(requiredFiles_out['higherOrders'])):
        # Even if nothing new is required, we still need to load existing files
        # MATLAB: After rehash, functions are available via eval(['@jacobian_',name])
        _loadExistingDerivativeFiles(sys, path, requiredFiles, requiredFiles_out)
        return
    
    # delete all existing files if setttings have changed
    # MATLAB: if deleteAll
    if deleteAll:
        # MATLAB: currDir = pwd;
        currDir = os.getcwd()
        # MATLAB: if isfolder(path)
        if os.path.isdir(path):
            # MATLAB: cd(path); delete *.m
            os.chdir(path)
            # Delete all .py files (instead of .m)
            for file in os.listdir('.'):
                if file.endswith('.py'):
                    try:
                        os.remove(file)
                    except:
                        pass
        # MATLAB: cd(currDir);
        os.chdir(currDir)
    
    # MATLAB: if options.verbose
    if options.get('verbose', False):
        # MATLAB: fprintf("Computing multivariate derivatives for dynamic '%s':\n", sys.name)
        print(f"Computing multivariate derivatives for dynamic '{sys.name}':")
    
    # read value for setting 'simplify' if provided, otherwise choose default
    # MATLAB: simplifyOpt = getDefaultValue('lagrangeRem.simplify',sys,struct(),options,'options');
    simplifyOpt = getDefaultValue('lagrangeRem.simplify', sys, {}, options, 'options')
    # MATLAB: if isfield(options,'lagrangeRem') && isfield(options.lagrangeRem,'simplify')
    if 'lagrangeRem' in options and 'simplify' in options.get('lagrangeRem', {}):
        # MATLAB: simplifyOpt = options.lagrangeRem.simplify;
        simplifyOpt = options['lagrangeRem']['simplify']
    # MATLAB: paramInt = [];
    paramInt = None
    # MATLAB: if isfield(options,'paramInt')
    if 'paramInt' in options:
        # MATLAB: paramInt = options.paramInt;
        paramInt = options['paramInt']
    
    # generate directory at path
    # MATLAB: if ~exist(path,'dir')
    if not os.path.exists(path):
        # MATLAB: mkdir(path);
        os.makedirs(path, exist_ok=True)
    # MATLAB: addpath(path);
    # In Python, we need to add path to sys.path so generated modules can be imported
    if path not in sys_module.path:
        sys_module.path.insert(0, path)
    
    # --- compute Jacobians ---------------------------------------------------
    # MATLAB: if options.verbose
    if options.get('verbose', False):
        # MATLAB: disp('  .. compute symbolic Jacobians');
        print('  .. compute symbolic Jacobians')
    
    # state/constraint equation
    # MATLAB: if requiredFiles.standard(1)
    if requiredFiles['standard'][0]:
        # MATLAB: Jdyn = aux_jacobians(fdyn,vars,simplifyOpt);
        Jdyn = aux_jacobians(fdyn, vars, simplifyOpt)
        # MATLAB: Jcon = aux_jacobians(fcon,vars,simplifyOpt);
        Jcon = aux_jacobians(fcon, vars, simplifyOpt)
        # MATLAB: [fp,Jp] = aux_parametric(fdyn,Jdyn,vars,paramInt);
        fp, Jp = aux_parametric(fdyn, Jdyn, vars, paramInt)
        
        # jacobian
        # MATLAB: fname = ['jacobian_' sys.name];
        fname = f'jacobian_{sys.name}'
        # MATLAB: if ~isempty(Jp)
        if Jp is not None and Jp.get('x') is not None:
            # Extract first slice (index 0) from 3D arrays for writeMatrixFile
            # MATLAB: writeMatrixFile({Jp.x,Jp.u},path,fname,...)
            # Note: Jp.x and Jp.u are 3D arrays, we extract the first slice
            Jp_x_2d = [[Jp['x'][i][j][0] for j in range(len(Jp['x'][0]))] for i in range(len(Jp['x']))]
            Jp_u_2d = [[Jp['u'][i][j][0] for j in range(len(Jp['u'][0]))] for i in range(len(Jp['u']))]
            jacobian_handle = writeMatrixFile([sp.Matrix(Jp_x_2d), sp.Matrix(Jp_u_2d)], path, fname,
                           'VarNamesIn', ['x', 'u', 'p'],
                           'VarNamesOut', ['A', 'B'],
                           'BracketSubs', True)
            # Attach jacobian to system object
            sys.jacobian = jacobian_handle
            # MATLAB: fname = ['parametricDynamicFile_' sys.name];
            fname = f'parametricDynamicFile_{sys.name}'
            # MATLAB: writeMatrixFile({fp},path,fname,...)
            # Extract first slice from fp (3D array)
            fp_2d = [[fp[i][0][0] for i in range(len(fp))]]
            writeMatrixFile([sp.Matrix(fp_2d)], path, fname,
                           'VarNamesIn', ['x', 'u'],
                           'VarNamesOut', ['f'],
                           'BracketSubs', True)
        # MATLAB: elseif ~isempty(vars.y)
        elif vars.get('y') is not None and (not hasattr(vars['y'], '__len__') or len(vars['y']) > 0):
            # MATLAB: writeMatrixFile({Jdyn.x,Jdyn.u,Jdyn.y,Jcon.x,Jcon.u,Jcon.y},path,fname,...)
            jacobian_handle = writeMatrixFile([Jdyn['x'], Jdyn['u'], Jdyn['y'],
                            Jcon['x'], Jcon['u'], Jcon['y']], path, fname,
                           'VarNamesIn', ['x', 'y', 'u'],
                           'VarNamesOut', ['A', 'B', 'C', 'D', 'E', 'F'],
                           'BracketSubs', True)
            # Attach jacobian to system object
            sys.jacobian = jacobian_handle
        else:
            # MATLAB: writeMatrixFile({Jdyn.x,Jdyn.u},path,fname,...)
            jacobian_handle = writeMatrixFile([Jdyn['x'], Jdyn['u']], path, fname,
                           'VarNamesIn', ['x', 'u'],
                           'VarNamesOut', ['A', 'B'],
                           'BracketSubs', True)
            # Attach jacobian to system object
            sys.jacobian = jacobian_handle
        
        # jacobian_freeParam
        # MATLAB: if ~isempty(vars.p)
        if vars.get('p') is not None and (not hasattr(vars['p'], '__len__') or len(vars['p']) > 0):
            # MATLAB: fname = ['jacobian_freeParam_' sys.name];
            fname = f'jacobian_freeParam_{sys.name}'
            # MATLAB: writeMatrixFile({Jdyn.x,Jdyn.u},path,fname,...)
            writeMatrixFile([Jdyn['x'], Jdyn['u']], path, fname,
                           'VarNamesIn', ['x', 'u', 'p'],
                           'VarNamesOut', ['A', 'B'],
                           'BracketSubs', True)
    
    # output equation
    # MATLAB: if requiredFiles_out.standard(1)
    if requiredFiles_out['standard'][0]:
        # MATLAB: Jout = aux_jacobians(fout,vars,simplifyOpt);
        Jout = aux_jacobians(fout, vars, simplifyOpt)
        # MATLAB: [fpout,Jpout] = aux_parametric(fout,Jout,vars,paramInt);
        fpout, Jpout = aux_parametric(fout, Jout, vars, paramInt)

        # out_jacobian
        # MATLAB: fname = ['out_jacobian_' sys.name];
        fname = f'out_jacobian_{sys.name}'
        # MATLAB: if ~isempty(Jpout)
        if Jpout is not None and Jpout.get('x') is not None:
            # Extract first slice (index 0) from 3D arrays for writeMatrixFile
            # MATLAB: writeMatrixFile({Jpout.x,Jpout.u},path,fname,...)
            Jpout_x_2d = [[Jpout['x'][i][j][0] for j in range(len(Jpout['x'][0]))] for i in range(len(Jpout['x']))]
            Jpout_u_2d = [[Jpout['u'][i][j][0] for j in range(len(Jpout['u'][0]))] for i in range(len(Jpout['u']))]
            writeMatrixFile([sp.Matrix(Jpout_x_2d), sp.Matrix(Jpout_u_2d)], path, fname,
                           'VarNamesIn', ['x', 'u', 'p'],
                           'VarNamesOut', ['C', 'D'],
                           'BracketSubs', True)
            # MATLAB: fname = ['out_parametricDynamicFile_' sys.name];
            fname = f'out_parametricDynamicFile_{sys.name}'
            # MATLAB: writeMatrixFile({fpout},path,fname,...)
            # Extract first slice from fpout
            fpout_2d = [[fpout[i][0][0] for i in range(len(fpout))]]
            writeMatrixFile([sp.Matrix(fpout_2d)], path, fname,
                           'VarNamesIn', ['x', 'u'],
                           'VarNamesOut', ['f'],
                           'BracketSubs', True)
        # MATLAB: elseif ~isempty(vars.y)
        elif vars.get('y') is not None and (not hasattr(vars['y'], '__len__') or len(vars['y']) > 0):
            # MATLAB: writeMatrixFile({Jout.x,Jout.u},path,fname,...)
            writeMatrixFile([Jout['x'], Jout['u']], path, fname,
                           'VarNamesIn', ['x', 'y', 'u'],
                           'VarNamesOut', ['C', 'D'],
                           'BracketSubs', True)
        else:
            # MATLAB: writeMatrixFile({Jout.x,Jout.u},path,fname,...)
            writeMatrixFile([Jout['x'], Jout['u']], path, fname,
                           'VarNamesIn', ['x', 'u'],
                           'VarNamesOut', ['C', 'D'],
                           'BracketSubs', True)
        
        # out_jacobian_freeParam
        # MATLAB: if ~isempty(vars.p)
        if vars.get('p') is not None and (not hasattr(vars['p'], '__len__') or len(vars['p']) > 0):
            # MATLAB: fname = ['out_jacobian_freeParam_' sys.name];
            fname = f'out_jacobian_freeParam_{sys.name}'
            # MATLAB: writeMatrixFile({Jout.x,Jout.u},path,fname,...)
            writeMatrixFile([Jout['x'], Jout['u']], path, fname,
                           'VarNamesIn', ['x', 'u', 'p'],
                           'VarNamesOut', ['C', 'D'],
                           'BracketSubs', True)

    # --- compute Hessians ----------------------------------------------------
    # MATLAB: if options.verbose
    if options.get('verbose', False):
        # MATLAB: disp('  .. compute symbolic Hessians');
        print('  .. compute symbolic Hessians')
    
    # state equation/constraint equation
    # MATLAB: if requiredFiles.standard(2) || requiredFiles.int(2)
    if requiredFiles['standard'][1] or requiredFiles['int'][1]:
        # MATLAB: J2dyn = aux_hessians(fdyn,vars,simplifyOpt);
        J2dyn = aux_hessians(fdyn, vars, simplifyOpt)
        # MATLAB: J2con = aux_hessians(fcon,vars,simplifyOpt);
        J2con = aux_hessians(fcon, vars, simplifyOpt)
    
    # with/without interval arithmetic
    # MATLAB: if requiredFiles.standard(2)
    if requiredFiles['standard'][1]:
        # MATLAB: fname = ['hessianTensor_' sys.name];
        fname = f'hessianTensor_{sys.name}'
        # MATLAB: writeHessianTensorFile(J2dyn,J2con,path,fname,vars,false,options);
        writeHessianTensorFile(J2dyn, J2con, path, fname, vars, False, options)
        # Load and attach hessian function handle
        hessian_file = os.path.join(path, f'{fname}.py')
        if os.path.isfile(hessian_file):
            import importlib.util
            module_name = f'{fname}_{id(sys)}'
            spec = importlib.util.spec_from_file_location(module_name, hessian_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.hessian = getattr(module, fname)
    # MATLAB: if requiredFiles.int(2)
    if requiredFiles['int'][1]:
        # MATLAB: fname = ['hessianTensorInt_' sys.name];
        fname = f'hessianTensorInt_{sys.name}'
        # MATLAB: writeHessianTensorFile(J2dyn,J2con,path,fname,vars,true,options);
        writeHessianTensorFile(J2dyn, J2con, path, fname, vars, True, options)
    
    # output equation
    # MATLAB: if requiredFiles_out.standard(2) || requiredFiles_out.int(2)
    if requiredFiles_out['standard'][1] or requiredFiles_out['int'][1]:
        # MATLAB: J2out = aux_hessians(fout,vars,simplifyOpt);
        J2out = aux_hessians(fout, vars, simplifyOpt)
    # with/without interval arithmetic
    # MATLAB: if requiredFiles_out.standard(2)
    if requiredFiles_out['standard'][1]:
        # MATLAB: fname = ['out_hessianTensor_' sys.name];
        fname = f'out_hessianTensor_{sys.name}'
        # MATLAB: writeHessianTensorFile(J2out,[],path,fname,vars,false,options);
        writeHessianTensorFile(J2out, [], path, fname, vars, False, options)
    # MATLAB: if requiredFiles_out.int(2)
    if requiredFiles_out['int'][1]:
        # MATLAB: fname = ['out_hessianTensorInt_' sys.name];
        fname = f'out_hessianTensorInt_{sys.name}'
        # MATLAB: writeHessianTensorFile(J2out,[],path,fname,vars,true,options);
        writeHessianTensorFile(J2out, [], path, fname, vars, True, options)

    # --- compute third-order derivatives -------------------------------------
    # MATLAB: if options.verbose
    if options.get('verbose', False):
        # MATLAB: disp('  .. compute symbolic third-order derivatives');
        print('  .. compute symbolic third-order derivatives')
    
    # state equation/constraint equation
    # MATLAB: if requiredFiles.standard(3) || requiredFiles.int(3)
    if requiredFiles['standard'][2] or requiredFiles['int'][2]:
        # MATLAB: J3dyn = aux_thirdOrderDerivatives(J2dyn,vars,simplifyOpt);
        J3dyn = aux_thirdOrderDerivatives(J2dyn, vars, simplifyOpt)
        # MATLAB: J3con = aux_thirdOrderDerivatives(J2con,vars,simplifyOpt);
        J3con = aux_thirdOrderDerivatives(J2con, vars, simplifyOpt)
    # with/without interval arithmetic
    # MATLAB: if requiredFiles.standard(3)
    if requiredFiles['standard'][2]:
        # MATLAB: fname = ['thirdOrderTensor_' sys.name];
        fname = f'thirdOrderTensor_{sys.name}'
        # MATLAB: write3rdOrderTensorFile(J3dyn,J3con,path,fname,vars,false,options);
        write3rdOrderTensorFile(J3dyn, J3con, path, fname, vars, False, options)
    # MATLAB: if requiredFiles.int(3)
    if requiredFiles['int'][2]:
        # MATLAB: fname = ['thirdOrderTensorInt_' sys.name];
        fname = f'thirdOrderTensorInt_{sys.name}'
        # MATLAB: write3rdOrderTensorFile(J3dyn,J3con,path,fname,vars,true,options);
        write3rdOrderTensorFile(J3dyn, J3con, path, fname, vars, True, options)
    
    # output equation
    # MATLAB: if requiredFiles_out.standard(3) || requiredFiles_out.int(3)
    if requiredFiles_out['standard'][2] or requiredFiles_out['int'][2]:
        # MATLAB: J3out = aux_thirdOrderDerivatives(J2out,vars,simplifyOpt);
        J3out = aux_thirdOrderDerivatives(J2out, vars, simplifyOpt)
    # with/without interval arithmetic
    # MATLAB: if requiredFiles_out.standard(3)
    if requiredFiles_out['standard'][2]:
        # MATLAB: fname = ['out_thirdOrderTensor_' sys.name];
        fname = f'out_thirdOrderTensor_{sys.name}'
        # MATLAB: write3rdOrderTensorFile(J3out,[],path,fname,vars,false,options);
        write3rdOrderTensorFile(J3out, [], path, fname, vars, False, options)
    # MATLAB: if requiredFiles_out.int(3)
    if requiredFiles_out['int'][2]:
        # MATLAB: fname = ['out_thirdOrderTensorInt_' sys.name];
        fname = f'out_thirdOrderTensorInt_{sys.name}'
        # MATLAB: write3rdOrderTensorFile(J3out,[],path,fname,vars,true,options);
        write3rdOrderTensorFile(J3out, [], path, fname, vars, True, options)

    # --- 4th and higher-order tensors (not for output equation) --------------
    # MATLAB: if any(requiredFiles.higherOrders)
    if np.any(requiredFiles['higherOrders']):
        # MATLAB: writeHigherOrderTensorFiles(fdyn,vars,varsDer,path,sys.name,options);
        writeHigherOrderTensorFiles(fdyn, vars, varsDer, path, sys.name, options)

    # rehash the folder so that new generated files are used
    # MATLAB: rehash path;
    # In Python, we need to load existing files that weren't regenerated
    # MATLAB: After rehash, functions are available via eval(['@jacobian_',name])
    # In Python, we need to explicitly import and attach them
    _loadExistingDerivativeFiles(sys, path, requiredFiles, requiredFiles_out)
    
    # save data so that symbolic computations do not have to be re-computed
    # MATLAB: filename = sprintf('%s%s%s_%s.mat',path,filesep,sys.name,'lastVersion');
    filename = os.path.join(path, f'{sys.name}_lastVersion.pkl')
    # MATLAB: save(filename,'storedData');
    with open(filename, 'wb') as f:
        pickle.dump({'storedData': storedData}, f)
    
    # log
    # MATLAB: if options.verbose
    if options.get('verbose', False):
        # MATLAB: fprintf("Done.\n")
        print("Done.")


# Auxiliary functions -----------------------------------------------------

def _loadExistingDerivativeFiles(sys: Any, path: str, requiredFiles: Dict[str, Any], 
                                  requiredFiles_out: Dict[str, Any]) -> None:
    """
    Load existing derivative files that weren't regenerated
    Equivalent to MATLAB's eval(['@jacobian_',name]) after rehash path
    
    Args:
        sys: contDynamics object
        path: path to folder where derivative files are stored
        requiredFiles: dict indicating which files need to be regenerated
        requiredFiles_out: dict indicating which output files need to be regenerated
    """
    import importlib.util
    
    # Load jacobian if file exists but wasn't regenerated
    # MATLAB: sys.jacobian = eval(['@jacobian_',sys.name]);
    if not requiredFiles['standard'][0]:  # File exists, wasn't regenerated
        jacobian_file = os.path.join(path, f'jacobian_{sys.name}.py')
        if os.path.isfile(jacobian_file):
            # Use a unique module name to avoid conflicts
            module_name = f'jacobian_{sys.name}_{id(sys)}'
            spec = importlib.util.spec_from_file_location(module_name, jacobian_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                jacobian_func = getattr(module, f'jacobian_{sys.name}')
                sys.jacobian = jacobian_func
    
    # Load hessian if file exists but wasn't regenerated
    if not requiredFiles['standard'][1]:
        hessian_file = os.path.join(path, f'hessianTensor_{sys.name}.py')
        if os.path.isfile(hessian_file):
            module_name = f'hessianTensor_{sys.name}_{id(sys)}'
            spec = importlib.util.spec_from_file_location(module_name, hessian_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.hessian = getattr(module, f'hessianTensor_{sys.name}')
    
    # Load thirdOrderTensor if file exists but wasn't regenerated
    if not requiredFiles['standard'][2]:
        tensor3_file = os.path.join(path, f'thirdOrderTensor_{sys.name}.py')
        if os.path.isfile(tensor3_file):
            module_name = f'thirdOrderTensor_{sys.name}_{id(sys)}'
            spec = importlib.util.spec_from_file_location(module_name, tensor3_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.thirdOrderTensor = getattr(module, f'thirdOrderTensor_{sys.name}')
    
    # Load output jacobian if file exists but wasn't regenerated
    if not requiredFiles_out['standard'][0]:
        out_jacobian_file = os.path.join(path, f'out_jacobian_{sys.name}.py')
        if os.path.isfile(out_jacobian_file):
            module_name = f'out_jacobian_{sys.name}_{id(sys)}'
            spec = importlib.util.spec_from_file_location(module_name, out_jacobian_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.out_jacobian = getattr(module, f'out_jacobian_{sys.name}')


def aux_defaultOptions(sys: Any) -> Dict[str, Any]:
    """
    Set required fields for default options
    
    Args:
        sys: contDynamics object
        
    Returns:
        options: dict with default options
    """
    
    # MATLAB: options = struct();
    options = {}
    
    # MATLAB: options.tensorOrder = getDefaultValue('tensorOrder',sys,struct(),options,'options');
    options['tensorOrder'] = getDefaultValue('tensorOrder', sys, {}, options, 'options')
    # MATLAB: options.tensorOrderOutput = getDefaultValue('tensorOrderOutput',sys,struct(),options,'options');
    options['tensorOrderOutput'] = getDefaultValue('tensorOrderOutput', sys, {}, options, 'options')
    
    # MATLAB: options.lagrangeRem.simplify = getDefaultValue('lagrangeRem.simplify',sys,struct(),options,'options');
    options['lagrangeRem'] = {}
    options['lagrangeRem']['simplify'] = getDefaultValue('lagrangeRem.simplify', sys, {}, options, 'options')
    # MATLAB: options.lagrangeRem.tensorParallel = getDefaultValue('lagrangeRem.tensorParallel',sys,struct(),options,'options');
    options['lagrangeRem']['tensorParallel'] = getDefaultValue('lagrangeRem.tensorParallel', sys, {}, options, 'options')
    # MATLAB: options.lagrangeRem.method = getDefaultValue('lagrangeRem.method',sys,struct(),options,'options');
    options['lagrangeRem']['method'] = getDefaultValue('lagrangeRem.method', sys, {}, options, 'options')
    # no .lagrangeRem.replacements
    # no .paramInt
    
    # MATLAB: options.verbose = false;
    options['verbose'] = False
    
    return options


def aux_insertSymVariables(sys: Any, vars: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """
    Insert symbolic variables into the system equations
    
    Args:
        sys: contDynamics object
        vars: dict with symbolic variables (keys: 'x', 'y', 'u', 'p')
        
    Returns:
        fdyn: symbolic differential equation
        fcon: symbolic constraint equation
        fout: symbolic output equation
    """
    
    # MATLAB: if isa(sys,'nonlinearARX')
    is_nonlinearARX = hasattr(sys, '__class__') and 'nonlinearARX' in sys.__class__.__name__.lower()
    
    if is_nonlinearARX:
        # MATLAB: fdyn = sys.mFile(vars.x,vars.u);
        fdyn = sys.mFile(vars['x'], vars['u'])
        # MATLAB: fcon = [];
        fcon = None
        # MATLAB: fout = [];
        fout = None
    # MATLAB: elseif isempty(vars.y) && isempty(vars.p)
    elif (vars.get('y') is None or (hasattr(vars['y'], '__len__') and len(vars['y']) == 0)) and \
         (vars.get('p') is None or (hasattr(vars['p'], '__len__') and len(vars['p']) == 0)):
        # class: nonlinearSys, nonlinearSysDT
        # MATLAB: fdyn = sys.mFile(vars.x,vars.u);
        fdyn = sys.mFile(vars['x'], vars['u'])
        # MATLAB: fcon = [];
        fcon = None
        # MATLAB: fout = sys.out_mFile(vars.x,vars.u);
        fout = sys.out_mFile(vars['x'], vars['u']) if hasattr(sys, 'out_mFile') else None
    # MATLAB: elseif isempty(vars.y) && ~isempty(vars.p)
    elif (vars.get('y') is None or (hasattr(vars['y'], '__len__') and len(vars['y']) == 0)) and \
         (vars.get('p') is not None and (not hasattr(vars['p'], '__len__') or len(vars['p']) > 0)):
        # class: nonlinParamSys
        # MATLAB: fdyn = sys.mFile(vars.x,vars.u,vars.p);
        fdyn = sys.mFile(vars['x'], vars['u'], vars['p'])
        # MATLAB: fcon = [];
        fcon = None
        # MATLAB: fout = sys.out_mFile(vars.x,vars.u,vars.p);
        fout = sys.out_mFile(vars['x'], vars['u'], vars['p']) if hasattr(sys, 'out_mFile') else None
    # MATLAB: elseif ~isempty(vars.y)
    elif vars.get('y') is not None and (not hasattr(vars['y'], '__len__') or len(vars['y']) > 0):
        # class: nonlinDASys
        # MATLAB: fdyn = sys.dynFile(vars.x,vars.y,vars.u);
        fdyn = sys.dynFile(vars['x'], vars['y'], vars['u'])
        # MATLAB: fcon = sys.conFile(vars.x,vars.y,vars.u);
        fcon = sys.conFile(vars['x'], vars['y'], vars['u'])
        # MATLAB: fout = sys.out_mFile(vars.x,vars.y,vars.u);
        fout = sys.out_mFile(vars['x'], vars['y'], vars['u']) if hasattr(sys, 'out_mFile') else None
    
    return fdyn, fcon, fout


def aux_jacobians(f: Any, vars: Dict[str, Any], simplifyOpt: str) -> Dict[str, Any]:
    """
    Compute symbolic Jacobians
    
    Args:
        f: symbolic differential equation
        vars: dict with symbolic variables (keys: 'x', 'u', 'y')
        simplifyOpt: specification if and how tensor should be simplified
        
    Returns:
        J: dict with Jacobians (keys: 'x', 'u', 'y')
    """
    
    # init
    # MATLAB: J = struct('x',[],'u',[],'y',[]);
    J = {'x': None, 'u': None, 'y': None}
    
    # Convert f to sympy Matrix if needed
    if f is None or (hasattr(f, '__len__') and len(f) == 0):
        J['x'] = sp.Matrix([])
        J['u'] = sp.Matrix([])
        J['y'] = sp.Matrix([])
        return J
    
    if not isinstance(f, sp.Matrix):
        if isinstance(f, (list, tuple, np.ndarray)):
            f = sp.Matrix(f)
        else:
            f = sp.Matrix([f])
    
    # Jacobians with respect to the state, input, and constraint state
    # MATLAB: J.x = jacobian(f,vars.x);
    if vars.get('x') is not None and (not hasattr(vars['x'], '__len__') or len(vars['x']) > 0):
        if isinstance(vars['x'], sp.Matrix):
            J['x'] = f.jacobian(vars['x'])
        else:
            J['x'] = f.jacobian(sp.Matrix(vars['x']))
    else:
        J['x'] = sp.Matrix([])
    
    # MATLAB: J.u = jacobian(f,vars.u);
    if vars.get('u') is not None and (not hasattr(vars['u'], '__len__') or len(vars['u']) > 0):
        if isinstance(vars['u'], sp.Matrix):
            J['u'] = f.jacobian(vars['u'])
        else:
            J['u'] = f.jacobian(sp.Matrix(vars['u']))
    else:
        J['u'] = sp.Matrix([])
    
    # MATLAB: J.y = jacobian(f,vars.y);
    if vars.get('y') is not None and (not hasattr(vars['y'], '__len__') or len(vars['y']) > 0):
        if isinstance(vars['y'], sp.Matrix):
            J['y'] = f.jacobian(vars['y'])
        else:
            J['y'] = f.jacobian(sp.Matrix(vars['y']))
    else:
        J['y'] = sp.Matrix([])
    
    # perform simplification
    # MATLAB: if strcmp(simplifyOpt,'simplify')
    if simplifyOpt == 'simplify':
        # MATLAB: J.x = simplify(J.x);
        J['x'] = sp.simplify(J['x'])
        # MATLAB: J.u = simplify(J.u);
        J['u'] = sp.simplify(J['u'])
        # MATLAB: J.y = simplify(J.y);
        J['y'] = sp.simplify(J['y'])
    # MATLAB: elseif strcmp(simplifyOpt,'collect')
    elif simplifyOpt == 'collect':
        # MATLAB: J.x = collect(J.x,vars.x);
        J['x'] = sp.collect(J['x'], vars.get('x'))
        # MATLAB: J.u = collect(J.u,vars.x);
        J['u'] = sp.collect(J['u'], vars.get('x'))
        # MATLAB: J.y = collect(J.y);
        J['y'] = sp.collect(J['y'])
    
    return J


def aux_parametric(f: Any, J: Dict[str, Any], vars: Dict[str, Any], paramInt: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Special function evaluation and derivatives for nonlinear
    dynamic/output equations with parameters linearly influencing the
    derivative
    
    Args:
        f: symbolic function
        J: dict with Jacobians (keys: 'x', 'u')
        vars: dict with symbolic variables (keys: 'x', 'u', 'p')
        paramInt: interval or vector for parameters
        
    Returns:
        fp: parametric function
        Jp: dict with parametric Jacobians (keys: 'x', 'u') or None
    """
    
    # MATLAB: if isempty(vars.p)
    if vars.get('p') is None or (hasattr(vars['p'], '__len__') and len(vars['p']) == 0):
        # MATLAB: fp = sym([]);
        fp = sp.Matrix([])
        # MATLAB: Jp = struct('x',{},'u',{});
        Jp = None
        return fp, Jp
    
    # number of states, parameters
    # MATLAB: numParams = numel(vars.p);
    if isinstance(vars['p'], sp.Matrix):
        numParams = vars['p'].shape[0]
    elif isinstance(vars['p'], (list, tuple, np.ndarray)):
        numParams = len(vars['p'])
    else:
        numParams = 1
    
    # Convert f to sympy Matrix if needed
    if not isinstance(f, sp.Matrix):
        if isinstance(f, (list, tuple, np.ndarray)):
            f = sp.Matrix(f)
        else:
            f = sp.Matrix([f])
    
    # insert parameters into dynamic file
    # MATLAB: fp = sym(zeros([size(f,1),1,numParams+1]));
    # In Python, we'll use a 3D list structure: [rows, 1, numParams+1]
    fp = [[[None for _ in range(numParams + 1)] for _ in range(1)] for _ in range(f.shape[0])]
    
    # store jacobians with respect to parameters
    # MATLAB: Jp = struct('x',[],'u',[]);
    Jp = {'x': None, 'u': None}
    # MATLAB: Jp.x = sym(zeros([size(J.x),numParams+1]));
    if J['x'] is not None and J['x'].shape[0] > 0 and J['x'].shape[1] > 0:
        # 3D array: [rows, cols, numParams+1]
        Jp['x'] = [[[None for _ in range(numParams + 1)] for _ in range(J['x'].shape[1])] for _ in range(J['x'].shape[0])]
    else:
        Jp['x'] = None
    # MATLAB: Jp.u = sym(zeros([size(J.u),numParams+1]));
    if J['u'] is not None and J['u'].shape[0] > 0 and J['u'].shape[1] > 0:
        # 3D array: [rows, cols, numParams+1]
        Jp['u'] = [[[None for _ in range(numParams + 1)] for _ in range(J['u'].shape[1])] for _ in range(J['u'].shape[0])]
    else:
        Jp['u'] = None
    
    try:
        # Convert vars.p to list for substitution
        if isinstance(vars['p'], sp.Matrix):
            vars_p_list = vars['p'].tolist()
        elif isinstance(vars['p'], (list, tuple)):
            vars_p_list = list(vars['p'])
        else:
            vars_p_list = [vars['p']]
        
        # part without parameters (-> insert zero for vars.p)
        # MATLAB: fp(:,:,1) = subs(f,vars.p,zeros(numParams,1));
        zero_sub = {vars_p_list[i]: 0 for i in range(numParams)}
        fp_0 = f.subs(zero_sub)
        for i in range(f.shape[0]):
            fp[i][0][0] = fp_0[i, 0]
        
        # MATLAB: Jp.x(:,:,1) = subs(J.x,vars.p,zeros(numParams,1));
        if J['x'] is not None:
            Jp_x_0 = J['x'].subs(zero_sub)
            for i in range(J['x'].shape[0]):
                for j in range(J['x'].shape[1]):
                    Jp['x'][i][j][0] = Jp_x_0[i, j]
        
        # MATLAB: Jp.u(:,:,1) = subs(J.u,vars.p,zeros(numParams,1));
        if J['u'] is not None:
            Jp_u_0 = J['u'].subs(zero_sub)
            for i in range(J['u'].shape[0]):
                for j in range(J['u'].shape[1]):
                    Jp['u'][i][j][0] = Jp_u_0[i, j]
        
        # part with parameters
        # MATLAB: for i=1:numParams
        for i in range(numParams):
            # MATLAB: fp(:,:,i+1) = subs(f,vars.p,unitvector(i,numParams)) - fp(:,:,1);
            unit_vec = unitvector(i + 1, numParams)  # unitvector uses 1-based indexing
            unit_sub = {vars_p_list[j]: float(unit_vec[j, 0]) for j in range(numParams)}
            fp_i = f.subs(unit_sub)
            for k in range(f.shape[0]):
                fp[k][0][i + 1] = fp_i[k, 0] - fp[k][0][0]
            
            # MATLAB: Jp.x(:,:,i+1) = subs(J.x,vars.p,unitvector(i,numParams)) - Jp.x(:,:,1);
            if J['x'] is not None:
                Jp_x_i = J['x'].subs(unit_sub)
                for k in range(J['x'].shape[0]):
                    for l in range(J['x'].shape[1]):
                        Jp['x'][k][l][i + 1] = Jp_x_i[k, l] - Jp['x'][k][l][0]
            
            # MATLAB: Jp.u(:,:,i+1) = subs(J.u,vars.p,unitvector(i,numParams)) - Jp.u(:,:,1);
            if J['u'] is not None:
                Jp_u_i = J['u'].subs(unit_sub)
                for k in range(J['u'].shape[0]):
                    for l in range(J['u'].shape[1]):
                        Jp['u'][k][l][i + 1] = Jp_u_i[k, l] - Jp['u'][k][l][0]

        # if parameters are uncertain within an interval
        # MATLAB: if isa(paramInt,'interval')
        if isinstance(paramInt, Interval):
            # normalize
            # MATLAB: pCenter = center(paramInt);
            pCenter = paramInt.center()
            # MATLAB: pDelta = rad(paramInt);
            pDelta = paramInt.rad()
        # MATLAB: elseif isnumeric(paramInt) && isvector(paramInt)
        elif isinstance(paramInt, (int, float, np.ndarray, list, tuple)) and \
             (isinstance(paramInt, np.ndarray) and paramInt.ndim == 1 or
              isinstance(paramInt, (list, tuple))):
            # MATLAB: pCenter = reshape(paramInt,[],1);
            pCenter = np.array(paramInt).reshape(-1, 1)
            # MATLAB: pDelta = zeros(numel(pCenter),1);
            pDelta = np.zeros((len(pCenter), 1))
        else:
            # MATLAB: throw(CORAerror('CORA:notSupported',...))
            from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
            raise CORAerror('CORA:notSupported',
                           'paramInt must be an interval or a double vector')

        # MATLAB: for i=1:numParams
        for i in range(numParams):
            # center
            # MATLAB: Jp.x(:,:,1) = Jp.x(:,:,1) + pCenter(i)*Jp.x(:,:,i+1);
            if Jp['x'] is not None:
                for k in range(len(Jp['x'])):
                    for l in range(len(Jp['x'][0])):
                        Jp['x'][k][l][0] = Jp['x'][k][l][0] + float(pCenter[i, 0]) * Jp['x'][k][l][i + 1]
            # MATLAB: Jp.u(:,:,1) = Jp.u(:,:,1) + pCenter(i)*Jp.u(:,:,i+1);
            if Jp['u'] is not None:
                for k in range(len(Jp['u'])):
                    for l in range(len(Jp['u'][0])):
                        Jp['u'][k][l][0] = Jp['u'][k][l][0] + float(pCenter[i, 0]) * Jp['u'][k][l][i + 1]
            # generators
            # MATLAB: Jp.x(:,:,i+1) = pDelta(i)*Jp.x(:,:,i+1);
            if Jp['x'] is not None:
                for k in range(len(Jp['x'])):
                    for l in range(len(Jp['x'][0])):
                        Jp['x'][k][l][i + 1] = float(pDelta[i, 0]) * Jp['x'][k][l][i + 1]
            # MATLAB: Jp.u(:,:,i+1) = pDelta(i)*Jp.u(:,:,i+1);
            if Jp['u'] is not None:
                for k in range(len(Jp['u'])):
                    for l in range(len(Jp['u'][0])):
                        Jp['u'][k][l][i + 1] = float(pDelta[i, 0]) * Jp['u'][k][l][i + 1]
        
        # Note: Keep Jp and fp as 3D structures
        # The first slice (index 0) will be extracted when passed to writeMatrixFile

    except Exception as e:
        # MATLAB: catch
        # MATLAB: Jp = struct('x',{},'u',{});
        Jp = None
        # MATLAB: disp('Parameters are not linearly influencing the system.');
        print('Parameters are not linearly influencing the system.')
        fp = sp.Matrix([])
    
    return fp, Jp


def aux_hessians(f: Any, vars: Dict[str, Any], simplifyOpt: str) -> Any:
    """
    Compute Hessian tensors for differential equation and
    constraint equation (only nonlinDASys)
    
    Args:
        f: symbolic equation
        vars: dict with symbolic variables (keys: 'x', 'y', 'u')
        simplifyOpt: specification if and how tensor should be simplified
        
    Returns:
        H: symbolic Hessian tensor (3D array)
    """
    
    # concatenate symbolic variables (with 'LR')
    # MATLAB: z = [vars.x;vars.y;vars.u];
    z_list = []
    if vars.get('x') is not None and (not hasattr(vars['x'], '__len__') or len(vars['x']) > 0):
        if isinstance(vars['x'], sp.Matrix):
            z_list.extend(vars['x'].tolist())
        else:
            z_list.extend(list(vars['x']))
    if vars.get('y') is not None and (not hasattr(vars['y'], '__len__') or len(vars['y']) > 0):
        if isinstance(vars['y'], sp.Matrix):
            z_list.extend(vars['y'].tolist())
        else:
            z_list.extend(list(vars['y']))
    if vars.get('u') is not None and (not hasattr(vars['u'], '__len__') or len(vars['u']) > 0):
        if isinstance(vars['u'], sp.Matrix):
            z_list.extend(vars['u'].tolist())
        else:
            z_list.extend(list(vars['u']))
    
    z = sp.Matrix(z_list) if z_list else sp.Matrix([])
    
    # Convert f to sympy Matrix if needed
    if f is None or (hasattr(f, '__len__') and len(f) == 0):
        return sp.Matrix([[[0]]])
    
    if not isinstance(f, sp.Matrix):
        if isinstance(f, (list, tuple, np.ndarray)):
            f = sp.Matrix(f)
        else:
            f = sp.Matrix([f])
    
    # potential simplifications
    # MATLAB: isSimplify = strcmp(simplifyOpt,'simplify');
    isSimplify = simplifyOpt == 'simplify'
    # MATLAB: isCollect = strcmp(simplifyOpt,'collect');
    isCollect = simplifyOpt == 'collect'
    
    # Jacobians and Hessians
    # MATLAB: J_comb = jacobian(f,z);
    J_comb = f.jacobian(z)
    # MATLAB: H = sym('x',[0,0,0]);
    # Initialize 3D tensor
    H_list = []
    # MATLAB: for k=1:size(J_comb,1)
    for k in range(J_comb.shape[0]):
        # compute Hessians/2nd-order Jacobians
        # MATLAB: H(k,:,:) = jacobian(J_comb(k,:),z);
        H_k = J_comb[k, :].jacobian(z)
        if isSimplify:
            # MATLAB: H(k,:,:) = simplify(H(k,:,:));
            H_k = sp.simplify(H_k)
        elif isCollect:
            # MATLAB: H(k,:,:) = collect(H(k,:,:),z);
            H_k = sp.collect(H_k, z)
        H_list.append(H_k)
    
    # Convert to 3D array representation
    # In MATLAB, this is a 3D array, but we'll keep it as a list of matrices
    # for easier handling
    return H_list


def aux_thirdOrderDerivatives(H: Any, vars: Dict[str, Any], simplifyOpt: str) -> Any:
    """
    Compute third-order derivatives for differential equation and
    constraint equation (only nonlinDASys)
    
    Args:
        H: symbolic Hessian tensor (list of matrices)
        vars: dict with symbolic variables (keys: 'x', 'y', 'u')
        simplifyOpt: specification if and how tensor should be simplified
        
    Returns:
        T: symbolic third-order tensor (4D array as list of lists of matrices)
    """
    
    # Convert H to list if needed
    if not isinstance(H, list):
        H = [H]
    
    if len(H) == 0:
        return []
    
    # MATLAB: [n,nrOfVars,~] = size(H);
    n = len(H)
    nrOfVars = H[0].shape[1] if isinstance(H[0], sp.Matrix) else H[0].shape[1]
    # MATLAB: T = sym(zeros(n,nrOfVars,nrOfVars,nrOfVars));
    T = [[[[None for _ in range(nrOfVars)] for _ in range(nrOfVars)] for _ in range(nrOfVars)] for _ in range(n)]
    
    # construct vector for which derivative is computed
    # MATLAB: z = [vars.x;vars.y;vars.u];
    z_list = []
    if vars.get('x') is not None and (not hasattr(vars['x'], '__len__') or len(vars['x']) > 0):
        if isinstance(vars['x'], sp.Matrix):
            z_list.extend(vars['x'].tolist())
        else:
            z_list.extend(list(vars['x']))
    if vars.get('y') is not None and (not hasattr(vars['y'], '__len__') or len(vars['y']) > 0):
        if isinstance(vars['y'], sp.Matrix):
            z_list.extend(vars['y'].tolist())
        else:
            z_list.extend(list(vars['y']))
    if vars.get('u') is not None and (not hasattr(vars['u'], '__len__') or len(vars['u']) > 0):
        if isinstance(vars['u'], sp.Matrix):
            z_list.extend(vars['u'].tolist())
        else:
            z_list.extend(list(vars['u']))
    
    z = sp.Matrix(z_list) if z_list else sp.Matrix([])
    
    # potential simplifications
    # MATLAB: isSimplify = strcmp(simplifyOpt,'simplify');
    isSimplify = simplifyOpt == 'simplify'
    # MATLAB: isCollect = strcmp(simplifyOpt,'collect');
    isCollect = simplifyOpt == 'collect'
    
    # compute third-order jacobians using 'LR' variables
    # MATLAB: for k=1:n
    for k in range(n):
        # MATLAB: for l=1:nrOfVars
        for l in range(nrOfVars):
            # compute 3rd-order Jacobians
            # MATLAB: if ~isempty(find(H(k,l,:), 1))
            # Check if H[k][l, :] has any non-zero elements
            H_kl = H[k][l, :] if isinstance(H[k], sp.Matrix) else H[k][l, :]
            if isinstance(H_kl, sp.Matrix):
                has_nonzero = any(H_kl[i] != 0 for i in range(H_kl.shape[0]))
            else:
                has_nonzero = np.any(H_kl != 0)
            
            if has_nonzero:
                # MATLAB: T(k,l,:,:) = jacobian(reshape(H(k,l,:),[nrOfVars,1]),z);
                H_kl_reshaped = H_kl.reshape(nrOfVars, 1) if isinstance(H_kl, sp.Matrix) else sp.Matrix(H_kl).reshape(nrOfVars, 1)
                T_kl = H_kl_reshaped.jacobian(z)
                if isSimplify:
                    # MATLAB: T(k,l,:,:) = simplify(T(k,l,:,:));
                    T_kl = sp.simplify(T_kl)
                elif isCollect:
                    # MATLAB: T(k,l,:,:) = collect(T(k,l,:,:),z);
                    T_kl = sp.collect(T_kl, z)
                # Store as 2D matrix
                T[k][l] = T_kl.tolist() if isinstance(T_kl, sp.Matrix) else T_kl
    
    return T

