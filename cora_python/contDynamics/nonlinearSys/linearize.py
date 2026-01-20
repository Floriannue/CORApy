"""
linearize - linearizes the nonlinear system; linearization error is not
   included yet

Syntax:
    [nlnsys,linsys,linParams,linOptions] = linearize(nlnsys,R,params,options)

Inputs:
    nlnsys - nonlinearSys object
    R - reachable set (only required if no linearization point given)
    params - model parameters
    options - options struct

Outputs:
    nlnsys - nonlinearSys object
    linsys - linearSys object, linParamSys object
    linParams - model parameter for the linearized system
    linOptions - options for the linearized system

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff
Written:       29-October-2007 
Last update:   22-January-2008
               29-June-2009
               04-August-2016
               15-August-2016
               12-September-2017
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Tuple, Any, Dict
from cora_python.contSet.zonotope import Zonotope


def linearize(nlnsys: Any, R: Any, params: Dict[str, Any], 
              options: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
    """
    Linearizes the nonlinear system; linearization error is not included yet
    
    Args:
        nlnsys: nonlinearSys object
        R: reachable set (only required if no linearization point given)
        params: model parameters (must contain 'uTrans' and 'U' keys)
        options: options struct (may contain 'linearizationPoint', 'refPoints', 'alg', 't', 'timeStep', 'Ronestep')
        
    Returns:
        nlnsys: nonlinearSys object (modified with linError)
        linsys: linearSys or linParamSys object
        linParams: model parameter for the linearized system (dict with 'U' and 'uTrans' keys)
        linOptions: options for the linearized system
    """
    
    # linearization point p.u of the input is the center of the input set
    # MATLAB: p.u = params.uTrans;
    p = {'u': params['uTrans']}
    
    # obtain linearization point
    # Initialize f0prev to None (will be set in else block if needed)
    f0prev = None
    
    # MATLAB: if isfield(options,'linearizationPoint')
    if 'linearizationPoint' in options:
        # MATLAB: p.x = options.linearizationPoint;
        p['x'] = options['linearizationPoint']
    # MATLAB: elseif isfield(options,'refPoints')
    elif 'refPoints' in options:
        # MATLAB: currentStep = round((options.t-params.tStart)/options.timeStep)+1;
        currentStep = int(round((options['t'] - params['tStart']) / options['timeStep']) + 1)
        # MATLAB: p.x = 1/2*sum(options.refPoints(:,currentStep:currentStep+1),2);
        refPoints = options['refPoints']
        p['x'] = 0.5 * np.sum(refPoints[:, currentStep-1:currentStep+1], axis=1, keepdims=True)
    else:
        # center of start set
        # MATLAB: cx = center(R);
        cx = R.center()
        
        # Check if center is too large (indicates set explosion before linReach can catch it)
        # This prevents overflow when computing f0prev
        if np.any(np.abs(cx) > 1e+100):
            from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
            raise CORAerror('CORA:reachSetExplosion')
        
        # linearization point p.x of the state is the center of the last
        # reachable set R translated by 0.5*delta_t*f0
        # MATLAB: f0prev = nlnsys.mFile(cx,p.u);
        f0prev = nlnsys.mFile(cx, p['u'])
        
        # MATLAB: try %if time step not yet created
        try:
            # MATLAB: p.x = cx + f0prev*0.5*options.timeStep;
            p['x'] = cx + f0prev * 0.5 * options['timeStep']
        except:
            # MATLAB: disp('time step not yet created');
            # MATLAB: p.x = cx;
            print('time step not yet created')
            p['x'] = cx
    
    # substitute p into the system equation to obtain the constant input
    # MATLAB: f0 = nlnsys.mFile(p.x,p.u);
    f0 = nlnsys.mFile(p['x'], p['u'])
    
    # substitute p into the Jacobian with respect to x and u to obtain the
    # system matrix A and the input matrix B
    # MATLAB: [A,B] = nlnsys.jacobian(p.x,p.u);
    A, B = nlnsys.jacobian(p['x'], p['u'])
    
    # MATLAB: linOptions=options;
    linOptions = options.copy()
    
    # MATLAB: if strcmp(options.alg,'linRem')
    if options['alg'] == 'linRem':
        # in order to compute dA,dB, we use the reachability set computed
        # for one step in initReach
        # MATLAB: [dA,dB] = lin_error2dAB(options.Ronestep,params.U,nlnsys.hessian,p);
        from cora_python.g.functions.helper.sets.contSet.contSet.lin_error2dAB import lin_error2dAB
        dA, dB = lin_error2dAB(options['Ronestep'], params['U'], nlnsys.hessian, p)
        
        # MATLAB: A = matZonotope(A,dA);
        from cora_python.matrixSet.matZonotope import matZonotope
        A = matZonotope(A, dA)
        
        # MATLAB: B = matZonotope(B,dB);
        B = matZonotope(B, dB)
        
        # MATLAB: linsys = linParamSys(A,1,'constParam');
        from cora_python.contDynamics.linearParamSys import LinearParamSys
        linsys = LinearParamSys(nlnsys.name + '_linearized', A, 1, 'constParam')
        
        # MATLAB: linOptions.compTimePoint = true;
        linOptions['compTimePoint'] = True
    else:
        # set up linearized system (B = 1 as input matrix encountered in
        # uncertain inputs)
        # MATLAB: linsys = linearSys([nlnsys.name '_linearized'],A,1);
        from cora_python.contDynamics.linearSys import LinearSys
        linsys = LinearSys(nlnsys.name + '_linearized', A, 1)
    
    # set up options for linearized system
    # MATLAB: linParams.U = B*(params.U+params.uTrans-p.u);
    linParams = {}
    # MATLAB: B*(params.U+params.uTrans-p.u) is matrix multiplication
    linParams['U'] = B @ (params['U'] + params['uTrans'] - p['u'])
    
    # MATLAB: Ucenter = center(linParams.U);
    Ucenter = linParams['U'].center()
    
    # MATLAB: linParams.U = linParams.U - Ucenter;
    linParams['U'] = linParams['U'] - Ucenter
    
    # MATLAB: linParams.uTrans = zonotope(f0 + Ucenter,zeros(size(f0,1),1));
    f0_shape = f0.shape[0] if f0.ndim > 0 else 1
    linParams['uTrans'] = Zonotope(f0 + Ucenter, np.zeros((f0_shape, 1)))
    
    # MATLAB: linOptions.originContained = false;
    linOptions['originContained'] = False
    
    # save constant input
    # MATLAB: nlnsys.linError.f0=f0;
    if nlnsys.linError is None:
        nlnsys.linError = type('obj', (object,), {})()
    nlnsys.linError.f0 = f0
    
    # save linearization point
    # MATLAB: nlnsys.linError.p=p;
    # Convert dict to object-like structure
    p_obj = type('obj', (object,), {})()
    p_obj.x = p['x']
    p_obj.u = p['u']
    nlnsys.linError.p = p_obj
    
    return nlnsys, linsys, linParams, linOptions

