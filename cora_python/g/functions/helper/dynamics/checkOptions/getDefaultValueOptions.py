"""
getDefaultValueOptions - contains list of default values for options

Syntax:
    defValue = getDefaultValueOptions(field,sys,params,options)

Inputs:
    field - struct field in params / options
    sys - object of system class
    params - struct containing model parameters
    options - struct containing algorithm parameters

Outputs:
    defValue - default value for given field

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: getDefaultValues

Authors:       Mark Wetzlinger
Written:       26-January-2021
Last update:   09-October-2023 (TL, split options/params)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict
from datetime import datetime
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.helper.dynamics.checkOptions.canUseParallelPool import canUseParallelPool


def getDefaultValueOptions(field: str, sys: Any, params: Dict[str, Any], 
                          options: Dict[str, Any]) -> Any:
    """
    Get default value for a field in options
    
    Args:
        field: struct field in options
        sys: object of system class
        params: struct containing model parameters
        options: struct containing algorithm parameters
        
    Returns:
        defValue: default value for given field
    """
    
    # search for default value in options.<field>
    # MATLAB: switch field
    if field == 'reductionTechnique':
        defValue = 'girard'
    elif field == 'reductionTechniqueUnderApprox':
        defValue = 'sum'
    elif field == 'linAlg':
        # linearization algorithm
        defValue = 'standard'
    elif field == 'zonotopeOrder':
        defValue = float('inf')
    elif field == 'reachAlg':
        defValue = 'standard'
    elif field == 'verbose':
        defValue = False
    elif field == 'reductionInterval':
        defValue = float('inf')
    
    # error bounds
    elif field == 'maxError':
        defValue = aux_def_maxError(sys, params, options)
    elif field == 'maxError_x':
        defValue = aux_def_maxError_x(sys, params, options)
    elif field == 'maxError_y':
        defValue = aux_def_maxError_y(sys, params, options)
    elif field == 'compTimePoint':
        defValue = True
    
    # polyZono
    elif field == 'polyZono.maxDepGenOrder':
        defValue = 20
    elif field == 'polyZono.maxPolyZonoRatio':
        defValue = float('inf')
    elif field == 'polyZono.restructureTechnique':
        defValue = 'reduceGirard'
    
    # lagrangeRem
    elif field == 'lagrangeRem.simplify':
        defValue = 'none'
    elif field == 'lagrangeRem.method':
        defValue = 'interval'
    elif field == 'lagrangeRem.tensorParallel':
        defValue = False
    elif field == 'lagrangeRem.optMethod':
        defValue = 'int'
    
    # ---
    elif field == 'contractor':
        defValue = 'linearize'
    elif field == 'iter':
        defValue = 2
    elif field == 'splits':
        defValue = 8
    elif field == 'orderInner':
        defValue = 5
    elif field == 'scaleFac':
        defValue = 'auto'
    elif field == 'type':
        defValue = 'standard'
    elif field == 'points':
        defValue = 10
    
    # frac
    elif field == 'fracVert':
        defValue = 0.5
    elif field == 'fracInpVert':
        defValue = 0.5
    elif field == 'nrConstInp':
        defValue = aux_def_nrConstInp(sys, params, options)
    elif field == 'p_conf':
        defValue = 0.8
    elif field == 'inpChanges':
        defValue = 1
    elif field == 'alg':
        # default algorithm depends on system and other parameters
        defValue = aux_def_alg(sys, params, options)
    elif field == 'tensorOrder':
        defValue = 2
    elif field == 'tensorOrderOutput':
        defValue = 2
    elif field == 'compOutputSet':
        defValue = True
    elif field == 'intersectInvariant':
        defValue = False
    elif field == 'timeStepDivider':
        defValue = 1
    elif field == 'postProcessingOrder':
        defValue = float('inf')
    elif field == 'armaxAlg':
        defValue = 'tvpGeneral'
    
    # for conform_white:
    elif field == 'cs.cp_lim':
        defValue = float('inf')
    elif field == 'cs.a_min':
        defValue = 0
    elif field == 'cs.a_max':
        defValue = float('inf')
    elif field == 'cs.cost':
        defValue = 'interval'
    elif field == 'cs.constraints':
        defValue = 'gen'
    elif field == 'cs.P':
        defValue = 1
    elif field == 'cs.w':
        defValue = aux_def_cs_w(sys, params, options)
    elif field == 'cs.verbose':
        defValue = False
    elif field == 'cs.robustnessMargin':
        defValue = 1e-9
    elif field == 'cs.derivRecomputation':
        defValue = True
    
    # for conform_black
    elif field == 'approx.p':
        defValue = 1
    elif field == 'approx.verbose':
        defValue = False
    elif field == 'approx.filename':
        # MATLAB: t = datetime; t.Format = 'yyyyMMddHHmmss'; defValue = sprintf('%s_approx', string(t));
        t = datetime.now()
        t_str = t.strftime('%Y%m%d%H%M%S')
        defValue = f'{t_str}_approx'
    elif field == 'approx.save_res':
        defValue = True
    
    # gp
    elif field == 'approx.gp_parallel':
        defValue = canUseParallelPool()
    elif field == 'approx.gp_runs':
        defValue = 1
    elif field == 'approx.gp_num_gen':
        defValue = 100
    elif field == 'approx.gp_pop_size':
        defValue = 300
    elif field == 'approx.gp_func_names':
        # MATLAB: defValue = {'times','minus','plus','tanh','square','sin','rdivide'};
        defValue = ['times', 'minus', 'plus', 'tanh', 'square', 'sin', 'rdivide']
    elif field == 'approx.gp_max_genes':
        defValue = 5
    elif field == 'approx.gp_max_depth':
        defValue = 6
    
    # cgp
    elif field == 'approx.cgp_num_gen':
        defValue = 5
    elif field == 'approx.cgp_n_m_conf':
        defValue = 5
    elif field == 'approx.cgp_pop_size_base':
        defValue = 10
    else:
        # MATLAB: throw(CORAerror('CORA:specialError',...))
        raise CORAerror('CORA:specialError', f"There is no default value for options.{field}.")
    
    return defValue


# Auxiliary functions -----------------------------------------------------

def aux_def_maxError(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default maxError
    """
    
    val = None
    # MATLAB: if isa(sys,'contDynamics') || isa(sys,'parallelHybridAutomaton')
    is_contDynamics = hasattr(sys, '__class__') and 'contDynamics' in sys.__class__.__name__.lower()
    is_parallelHybridAutomaton = hasattr(sys, '__class__') and 'parallelHybridAutomaton' in sys.__class__.__name__.lower()
    
    if is_contDynamics or is_parallelHybridAutomaton:
        # MATLAB: val = Inf(sys.nrOfDims,1);
        val = np.full((sys.nrOfDims, 1), float('inf'))
    # MATLAB: elseif isa(sys,'hybridAutomaton')
    elif hasattr(sys, '__class__') and 'hybridAutomaton' in sys.__class__.__name__.lower():
        # MATLAB: n = sys.nrOfDims;
        n = sys.nrOfDims
        # MATLAB: if all(n(1) == n)
        if np.all(n[0] == n):
            # MATLAB: val = Inf(n(1),1);
            val = np.full((n[0], 1), float('inf'))
        else:
            # MATLAB: throw(CORAerror('CORA:notSupported',...))
            raise CORAerror('CORA:notSupported',
                          'Default value for maxError not supported for hybrid automata with varying number of states per location.')
    # no assignment for hybridAutomaton / parallelHybridAutomaton
    
    return val


def aux_def_nrConstInp(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default nrConstInp
    """
    
    val = None
    # MATLAB: if isa(sys,'contDynamics')
    if hasattr(sys, '__class__') and 'contDynamics' in sys.__class__.__name__.lower():
        # MATLAB: if isa(sys,'linearSysDT') || isa(sys,'nonlinearSysDT') || isa(sys,'neurNetContrSys')
        is_linearSysDT = hasattr(sys, '__class__') and 'linearSysDT' in sys.__class__.__name__.lower()
        is_nonlinearSysDT = hasattr(sys, '__class__') and 'nonlinearSysDT' in sys.__class__.__name__.lower()
        is_neurNetContrSys = hasattr(sys, '__class__') and 'neurNetContrSys' in sys.__class__.__name__.lower()
        
        if is_linearSysDT or is_nonlinearSysDT or is_neurNetContrSys:
            # MATLAB: steps = round((params.tFinal - params.tStart) / sys.dt);
            steps = round((params['tFinal'] - params['tStart']) / sys.dt)
            
            # MATLAB: if isinf(steps)
            if np.isinf(steps):
                val = 1
            else:
                # start at 10, go down to 1
                # MATLAB: for i=10:-1:1
                for i in range(10, 0, -1):
                    # MATLAB: if mod(steps,i) == 0
                    if steps % i == 0:
                        val = i
                        break
        else:
            # MATLAB: if size(params.u,2) > 1
            u = params.get('u', None)
            if u is not None and isinstance(u, np.ndarray) and u.shape[1] > 1:
                # MATLAB: if isa(sys,'linearSys') && any(any(sys.D))
                is_linearSys = hasattr(sys, '__class__') and 'linearSys' in sys.__class__.__name__.lower()
                if is_linearSys and hasattr(sys, 'D') and np.any(sys.D):
                    # MATLAB: val = size(params.u,2) - 1;
                    val = u.shape[1] - 1
                else:
                    # MATLAB: val = size(params.u,2);
                    val = u.shape[1]
            else:  # no input trajectory
                val = 10
    # MATLAB: elseif isa(sys,'hybridAutomaton') || isa(sys,'parallelHybridAutomaton')
    elif (hasattr(sys, '__class__') and 
          ('hybridAutomaton' in sys.__class__.__name__.lower() or 
           'parallelHybridAutomaton' in sys.__class__.__name__.lower())):
        # this will most likely be changed in the future (e.g., different
        # values for each location)
        val = 10
    
    return val


def aux_def_maxError_x(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default maxError_x (only DA systems)
    """
    
    val = None
    # MATLAB: if isa(sys,'nonlinDASys')
    if hasattr(sys, '__class__') and 'nonlinDASys' in sys.__class__.__name__.lower():
        # MATLAB: val = Inf(sys.nrOfDims,1);
        val = np.full((sys.nrOfDims, 1), float('inf'))
    
    return val


def aux_def_maxError_y(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Auxiliary function to get default maxError_y (only DA systems)
    """
    
    val = None
    # MATLAB: if isa(sys,'nonlinDASys')
    if hasattr(sys, '__class__') and 'nonlinDASys' in sys.__class__.__name__.lower():
        # MATLAB: val = Inf(sys.nrOfConstraints,1);
        val = np.full((sys.nrOfConstraints, 1), float('inf'))
    
    return val


def aux_def_alg(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> str:
    """
    Auxiliary function to get default algorithm
    """
    
    # default
    val = 'lin'
    
    # explicitly stated for the below classes
    # MATLAB: if isa(sys,'nonlinearSysDT')
    if hasattr(sys, '__class__') and 'nonlinearSysDT' in sys.__class__.__name__.lower():
        val = 'lin'
    # MATLAB: elseif isa(sys,'nonlinDASys')
    elif hasattr(sys, '__class__') and 'nonlinDASys' in sys.__class__.__name__.lower():
        val = 'lin'
    
    return val


def aux_def_cs_w(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> np.ndarray:
    """
    Auxiliary function to get default cs.w
    """
    
    # MATLAB: if isprop(sys,'dt')
    if hasattr(sys, 'dt'):
        dt = sys.dt
    else:  # get dt from testsuite
        # MATLAB: dt = params.testSuite{1}.sampleTime;
        dt = params['testSuite'][0]['sampleTime']
    
    # MATLAB: maxNrOfTimeSteps = ceil(round(params.tFinal/dt,2));
    maxNrOfTimeSteps = int(np.ceil(round(params['tFinal'] / dt, 2)))
    # MATLAB: val = ones(1,maxNrOfTimeSteps+1);
    val = np.ones((1, maxNrOfTimeSteps + 1))
    
    return val

