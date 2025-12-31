"""
priv_checkTensorRecomputation - checks whether symbolic computations have to
   be performed or whether the old derivations have remained unchanged

Syntax:
    [generateFiles,requiredFiles_out,requiredData] = ...
       priv_checkTensorRecomputation(sys,fdyn,fcon,fout,path,options)

Inputs:
    sys - contDynamics object
    fdyn - symbolic differential equation
    fcon - symbolic constraint equation (only nonlinDASys)
    fout - symbolic output equation
    path - path to folder where tensor files are computed
    options - options for reachability analysis
           .tensorOrder
           .lagrangeRem

Outputs:
    requiredFiles - logical array which tensor files need to be recomputed
                    (for the dynamic/constaint equations)
    requiredFiles_out - logical array which tensor files need to be
                        recomputed (for the output equation)
    requiredData - information about settings for tensor file generation
    deleteAll - true/false whether all files should be deleted

Example:
    -

Authors:       Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
Written:       ---
Last update:   01-February-2021 (MW, introduction of requiredFiles)
               18-November-2022 (MW, integrate output equation)
Last revision: 07-October-2024 (MW, complete refactor)
               Automatic python translation: Florian Nüssel BA 2025
"""

import os
import numpy as np
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
from cora_python.g.macros.CORAVERSION import CORAVERSION
from cora_python.g.functions.helper.dynamics.checkOptions.getDefaultValue import getDefaultValue
from cora_python.g.functions.matlab.function_handle.isequalFunctionHandle import isequalFunctionHandle
from cora_python.g.functions.matlab.struct.rmiffield import rmiffield


def priv_checkTensorRecomputation(sys: Any, fdyn: Any, fcon: Any, fout: Any,
                                   path: str, options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], bool]:
    """
    Checks whether symbolic computations have to be performed or whether the old derivations have remained unchanged
    
    Args:
        sys: contDynamics object
        fdyn: symbolic differential equation
        fcon: symbolic constraint equation (only nonlinDASys)
        fout: symbolic output equation
        path: path to folder where tensor files are computed
        options: options for reachability analysis (must contain 'tensorOrder' and optionally 'lagrangeRem')
        
    Returns:
        requiredFiles: dict with logical arrays indicating which tensor files need to be recomputed
                      (for the dynamic/constraint equations)
        requiredFiles_out: dict with logical arrays indicating which tensor files need to be recomputed
                          (for the output equation)
        requiredData: information about settings for tensor file generation
        deleteAll: True/False whether all files should be deleted
    """
    
    # list required files for dynamic/constraint equation and output equation
    # given the current dynamics and tensorOrder
    # MATLAB: requiredFiles = aux_listRequiredFiles(sys,true,options);
    requiredFiles = aux_listRequiredFiles(sys, True, options)
    # MATLAB: requiredFiles_out = aux_listRequiredFiles(sys,false,options);
    requiredFiles_out = aux_listRequiredFiles(sys, False, options)
    # MATLAB: deleteAll = false;
    deleteAll = False
    
    # check if expected path to .mat-file containing information about the 
    # folder content exists (true if derivatives previously computed); if yes,
    # try loading stored data from that file
    # MATLAB: try
    storedData = {}
    try:
        # MATLAB: load([path filesep sys.name '_lastVersion.mat'],'storedData');
        import pickle
        mat_file_path = os.path.join(path, f"{sys.name}_lastVersion.pkl")
        if os.path.exists(mat_file_path):
            with open(mat_file_path, 'rb') as f:
                stored_data_dict = pickle.load(f)
                storedData = stored_data_dict.get('storedData', {})
    except Exception:
        # MATLAB: storedData = struct([]);
        storedData = {}
    
    # generate equivalent struct to storedData with current settings
    # MATLAB: requiredData = aux_requiredData(sys,fdyn,fcon,fout,options);
    requiredData = aux_requiredData(sys, fdyn, fcon, fout, options)
    
    # also save current CORAversion and datetime of generation
    # MATLAB: requiredData.CORAversion = CORAVERSION;
    requiredData['CORAversion'] = CORAVERSION()
    # MATLAB: requiredData.timeStamp = datetime;
    requiredData['timeStamp'] = datetime.now()
    
    # structs must be equal, otherwise we must not check for files that are
    # computed already (see below) since we have different dynamics/settings!
    # MATLAB: if isempty(storedData) || ~aux_compareData(storedData,requiredData)
    if not storedData or not aux_compareData(storedData, requiredData):
        # MATLAB: deleteAll = true;
        deleteAll = True
        # MATLAB: return;
        return requiredFiles, requiredFiles_out, requiredData, deleteAll
    
    # update required files: remove those that are computed already
    # MATLAB: [requiredFiles,requiredFiles_out] = aux_updateRequiredFiles(sys,path,...
    #    requiredFiles,requiredFiles_out);
    requiredFiles, requiredFiles_out = aux_updateRequiredFiles(sys, path,
                                                              requiredFiles, requiredFiles_out)
    
    return requiredFiles, requiredFiles_out, requiredData, deleteAll


# Auxiliary functions -----------------------------------------------------

def aux_compareData(S1: Dict[str, Any], S2: Dict[str, Any]) -> bool:
    """
    Compares the data between the two structs; normally, isequal(S1,S2)
    should suffice, but .lagrangeRem.replacements is a function handle, for
    which the following behavior holds true:
      f = lambda x: x[0]*x[1]
      g = lambda x: x[0]*x[1]
      assert(f != g)  # <-- this is a problem for us
    hence, we compare the struct while omitting any occurrences of the field
    .lagrangeRem.replacements and compare them separately
    """
    
    # both must have .lagrangeRem -> check lagrangeRem.replacements separately
    # MATLAB: if xor(isfield(S1.lagrangeRem,'replacements'),isfield(S2.lagrangeRem,'replacements'))
    has_replacements_S1 = 'lagrangeRem' in S1 and 'replacements' in S1.get('lagrangeRem', {})
    has_replacements_S2 = 'lagrangeRem' in S2 and 'replacements' in S2.get('lagrangeRem', {})
    
    if has_replacements_S1 != has_replacements_S2:
        return False
    
    if has_replacements_S1 and has_replacements_S2:
        # MATLAB: if ~isequalFunctionHandle(S1.lagrangeRem.replacements,S2.lagrangeRem.replacements)
        if not isequalFunctionHandle(S1['lagrangeRem']['replacements'], S2['lagrangeRem']['replacements']):
            return False
        # remove replacements from further checking
        # MATLAB: S1.lagrangeRem = rmiffield(S1.lagrangeRem,'replacements');
        S1_copy = S1.copy()
        S1_copy['lagrangeRem'] = rmiffield(S1['lagrangeRem'], 'replacements')
        S1 = S1_copy
        # MATLAB: S2.lagrangeRem = rmiffield(S2.lagrangeRem,'replacements');
        S2_copy = S2.copy()
        S2_copy['lagrangeRem'] = rmiffield(S2['lagrangeRem'], 'replacements')
        S2 = S2_copy
    
    # unfortunately, we also need to check paramInt separately... the reason is
    # that the isequal-call below for the struct (correctly!) uses the
    # isequal-function for interval or double. In this case, a comparison of
    # interval(a,a) and a would return true (as intended). However, we must
    # ensure that both .paramInt are of the same class since the computation of
    # derivatives differs whether .paramInt is an interval or a double.
    # MATLAB: if xor(isfield(S1.lagrangeRem,'replacements'),isfield(S2.lagrangeRem,'replacements'))
    # NOTE: This check seems to be duplicated in MATLAB code - checking paramInt instead
    has_paramInt_S1 = 'paramInt' in S1
    has_paramInt_S2 = 'paramInt' in S2
    
    if has_paramInt_S1 != has_paramInt_S2:
        return False
    
    # MATLAB: if isfield(S1,'paramInt') && isfield(S2,'paramInt') ...
    #        && xor(isa(S1.paramInt,'interval'),isa(S2.paramInt,'interval'))
    if has_paramInt_S1 and has_paramInt_S2:
        from cora_python.contSet.interval import Interval
        is_interval_S1 = isinstance(S1['paramInt'], Interval)
        is_interval_S2 = isinstance(S2['paramInt'], Interval)
        if is_interval_S1 != is_interval_S2:
            return False
    
    # remove version and generated as they do not need to be equal
    # MATLAB: S1 = rmiffield(S1,{'CORAversion','timeStamp'});
    S1 = rmiffield(S1, ['CORAversion', 'timeStamp'])
    # MATLAB: S2 = rmiffield(S2,{'CORAversion','timeStamp'});
    S2 = rmiffield(S2, ['CORAversion', 'timeStamp'])
    
    # use built-in broadcast of isequal functions via struct comparison
    # MATLAB: res = isequal(S1,S2);
    # In Python, we need to recursively compare dictionaries
    res = _dict_equal(S1, S2)
    
    return res


def _dict_equal(d1: Dict[str, Any], d2: Dict[str, Any]) -> bool:
    """
    Recursively compare two dictionaries for equality
    Handles nested dicts, lists, numpy arrays, and other types
    """
    if set(d1.keys()) != set(d2.keys()):
        return False
    
    for key in d1.keys():
        val1 = d1[key]
        val2 = d2[key]
        
        if isinstance(val1, dict) and isinstance(val2, dict):
            if not _dict_equal(val1, val2):
                return False
        elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if len(val1) != len(val2):
                return False
            for v1, v2 in zip(val1, val2):
                if isinstance(v1, dict) and isinstance(v2, dict):
                    if not _dict_equal(v1, v2):
                        return False
                elif v1 != v2:
                    return False
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False
        else:
            if val1 != val2:
                return False
    
    return True


def aux_requiredData(sys: Any, fdyn: Any, fcon: Any, fout: Any, 
                     options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate equivalent struct to storedData with current settings
    """
    
    # save dynamics/constraint/output equation
    # MATLAB: requiredData.fdyn = fdyn;
    requiredData = {'fdyn': fdyn}
    # MATLAB: requiredData.fcon = fcon;
    requiredData['fcon'] = fcon
    # MATLAB: requiredData.fout = fout;
    requiredData['fout'] = fout
    
    # save settings for Lagrange remainder: here, we are a bit lenient with the
    # caller... add default values for all non-set fields
    # MATLAB: if ~isfield(options,'lagrangeRem')
    if 'lagrangeRem' not in options:
        # MATLAB: requiredData.lagrangeRem.simplify = getDefaultValue('lagrangeRem.simplify',sys,struct(),options,'options');
        requiredData['lagrangeRem'] = {}
        requiredData['lagrangeRem']['simplify'] = getDefaultValue('lagrangeRem.simplify', sys, {}, options, 'options')
        # MATLAB: requiredData.lagrangeRem.tensorParallel = getDefaultValue('lagrangeRem.tensorParallel',sys,struct(),options,'options');
        requiredData['lagrangeRem']['tensorParallel'] = getDefaultValue('lagrangeRem.tensorParallel', sys, {}, options, 'options')
        # MATLAB: requiredData.lagrangeRem.method = getDefaultValue('lagrangeRem.method',sys,struct(),options,'options');
        requiredData['lagrangeRem']['method'] = getDefaultValue('lagrangeRem.method', sys, {}, options, 'options')
        # no .replacements here
    else:
        # MATLAB: requiredData.lagrangeRem = options.lagrangeRem;
        requiredData['lagrangeRem'] = options['lagrangeRem'].copy() if isinstance(options['lagrangeRem'], dict) else options['lagrangeRem']
        # MATLAB: if ~isfield(options.lagrangeRem,'simplify')
        if 'simplify' not in options['lagrangeRem']:
            # MATLAB: requiredData.lagrangeRem.simplify = getDefaultValue('lagrangeRem.simplify',sys,struct(),options,'options');
            requiredData['lagrangeRem']['simplify'] = getDefaultValue('lagrangeRem.simplify', sys, {}, options, 'options')
        # MATLAB: if ~isfield(options.lagrangeRem,'tensorParallel')
        if 'tensorParallel' not in options['lagrangeRem']:
            # MATLAB: requiredData.lagrangeRem.tensorParallel = getDefaultValue('lagrangeRem.tensorParallel',sys,struct(),options,'options');
            requiredData['lagrangeRem']['tensorParallel'] = getDefaultValue('lagrangeRem.tensorParallel', sys, {}, options, 'options')
        # MATLAB: if ~isfield(options.lagrangeRem,'method')
        if 'method' not in options['lagrangeRem']:
            # MATLAB: requiredData.lagrangeRem.method = getDefaultValue('lagrangeRem.method',sys,struct(),options,'options');
            requiredData['lagrangeRem']['method'] = getDefaultValue('lagrangeRem.method', sys, {}, options, 'options')
        # no .replacements here...
    
    # only for dynamical systems with parameters
    # MATLAB: if isfield(options,'paramInt')
    if 'paramInt' in options:
        # MATLAB: requiredData.paramInt = options.paramInt;
        requiredData['paramInt'] = options['paramInt']
    
    return requiredData


def aux_listRequiredFiles(sys: Any, isState: bool, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    List the files that need to be generated according to the given settings
      isState == True    for state equation (differential equation)
      isState == False   for output equation
    """
    
    # initial assumption: state which files are required
    # MATLAB: files.higherOrders = false(10,1);
    files = {'higherOrders': np.zeros(10, dtype=bool)}
    # without interval arithmetic
    # MATLAB: files.standard = false(3,1);
    files['standard'] = np.zeros(3, dtype=bool)
    # with interval arithmetic
    # MATLAB: files.int = false(3,1);
    files['int'] = np.zeros(3, dtype=bool)
    
    # shortcut if we deal with the output equation
    # MATLAB: if ~isState
    if not isState:
        # MATLAB: if isa(sys,'nonlinearARX')
        is_nonlinearARX = hasattr(sys, '__class__') and 'nonlinearARX' in sys.__class__.__name__.lower()
        if is_nonlinearARX:
            # no output files required
            # MATLAB: return
            return files
        # MATLAB: elseif all(sys.out_isLinear)
        elif hasattr(sys, 'out_isLinear') and np.all(sys.out_isLinear):
            # only Jacobian required... all further derivatives are zero
            # MATLAB: files.standard(1) = true;
            files['standard'][0] = True
            # MATLAB: return
            return files
    
    # conformant synthesis, no output equation
    # MATLAB: if isfield(options,'cs')
    if 'cs' in options:
        # MATLAB: files.standard(1) = true;
        files['standard'][0] = True
        # MATLAB: return
        return files
    
    # adaptive algorithm... requires multiple no-interval-arithmetic and
    # interval arithmetic tensors
    # MATLAB: if ( isa(sys,'nonlinearSys') || isa(sys,'nonlinDASys') || isa(sys,'nonlinearSysDT') ) ...
    #        && isfield(options,'alg') && contains(options.alg,'adaptive')
    is_nonlinearSys = hasattr(sys, '__class__') and 'nonlinearSys' in sys.__class__.__name__.lower()
    is_nonlinDASys = hasattr(sys, '__class__') and 'nonlinDASys' in sys.__class__.__name__.lower()
    is_nonlinearSysDT = hasattr(sys, '__class__') and 'nonlinearSysDT' in sys.__class__.__name__.lower()
    
    if (is_nonlinearSys or is_nonlinDASys or is_nonlinearSysDT) and 'alg' in options and 'adaptive' in str(options['alg']):
        # only nonlinearSys has 'lin' and 'poly' in options.alg
        # MATLAB: if contains(options.alg,'lin')
        if 'lin' in str(options['alg']):
            # MATLAB: files.standard(1:2) = true;
            files['standard'][0:2] = True
            # MATLAB: files.int(2:3) = true;
            files['int'][1:3] = True
        else:  # 'poly'
            # MATLAB: files.standard(1:2) = true;
            files['standard'][0:2] = True
            # MATLAB: files.int(3) = true;
            files['int'][2] = True
        # MATLAB: return
        return files
    
    # read out tensor order
    # MATLAB: if isState
    if isState:
        # MATLAB: tensorOrder = options.tensorOrder;
        tensorOrder = options['tensorOrder']
    else:
        # MATLAB: tensorOrder = options.tensorOrderOutput;
        tensorOrder = options['tensorOrderOutput']
    
    # always compute Jacobian...
    # MATLAB: files.standard(1) = true;
    files['standard'][0] = True
    # standard requirements for derivatives computation
    # MATLAB: if tensorOrder <= 3
    if tensorOrder <= 3:
        # MATLAB: files.standard(1:tensorOrder-1) = true;
        files['standard'][0:tensorOrder-1] = True
        # MATLAB: files.int(tensorOrder) = true;
        files['int'][tensorOrder-1] = True
    else:
        # MATLAB: files.standard(1:3) = true;
        files['standard'][0:3] = True
        # MATLAB: files.higherOrders(4:tensorOrder) = true;
        files['higherOrders'][3:tensorOrder] = True
    
    return files


def aux_updateRequiredFiles(sys: Any, path: str, requiredFiles: Dict[str, Any],
                            requiredFiles_out: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Update required files: remove those that are computed already
    """
    
    # (hardcoded) names of tensor files
    # MATLAB: prefixes = {'jacobian','hessianTensor','thirdOrderTensor'};
    prefixes = ['jacobian', 'hessianTensor', 'thirdOrderTensor']
    
    # go through files in path and find which ones are there already...
    # MATLAB: for i=1:3
    for i in range(3):
        # dynamic equation
        # MATLAB: requiredFiles.standard(i) = requiredFiles.standard(i) ...
        #        && ~isfile([path filesep prefixes{i} '_' sys.name '.m']);
        file_path_std = os.path.join(path, f"{prefixes[i]}_{sys.name}.py")
        requiredFiles['standard'][i] = requiredFiles['standard'][i] and not os.path.isfile(file_path_std)
        
        # MATLAB: requiredFiles.int(i) = requiredFiles.int(i) ...
        #        && ~isfile([path filesep prefixes{i} 'Int_' sys.name '.m']);
        file_path_int = os.path.join(path, f"{prefixes[i]}Int_{sys.name}.py")
        requiredFiles['int'][i] = requiredFiles['int'][i] and not os.path.isfile(file_path_int)
        
        # output equation
        # MATLAB: requiredFiles_out.standard(i) = requiredFiles_out.standard(i) ...
        #        && ~isfile([path filesep 'out_' prefixes{i} '_' sys.name '.m']);
        file_path_out_std = os.path.join(path, f"out_{prefixes[i]}_{sys.name}.py")
        requiredFiles_out['standard'][i] = requiredFiles_out['standard'][i] and not os.path.isfile(file_path_out_std)
        
        # MATLAB: requiredFiles_out.int(i) = requiredFiles_out.int(i) ...
        #        && ~isfile([path filesep 'out_' prefixes{i} 'Int_' sys.name '.m']);
        file_path_out_int = os.path.join(path, f"out_{prefixes[i]}Int_{sys.name}.py")
        requiredFiles_out['int'][i] = requiredFiles_out['int'][i] and not os.path.isfile(file_path_out_int)
    
    # no separation of file names (with Int) for 4th or higher orders
    # MATLAB: for i=4:10
    for i in range(4, 11):
        # MATLAB: prefixes{i} = ['tensor' num2str(i)];
        prefix = f'tensor{i}'
        # MATLAB: requiredFiles.higherOrders(i) = requiredFiles.higherOrders(i) ...
        #        && ~isfile([path filesep prefixes{i} '_' sys.name '.m']);
        file_path = os.path.join(path, f"{prefix}_{sys.name}.py")
        requiredFiles['higherOrders'][i-1] = requiredFiles['higherOrders'][i-1] and not os.path.isfile(file_path)
    
    return requiredFiles, requiredFiles_out

