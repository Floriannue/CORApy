% Check what properties the linParamSys has after linearize
% Add CORA to path if needed
if ~exist('zonotope', 'file')
    addpath(genpath('cora_matlab'));
end

fprintf('=== Checking linParamSys properties after linearize ===\n\n');

% Create a simple 2D nonlinear parametric system
f = @(x, u, p) [x(2); -x(1) - p(1)*x(2) + u(1)];
sys = nonlinParamSys('simple_oscillator', f);

% Parameters
params.R0 = zonotope([0; 0], [0.1, 0; 0, 0.1]);
params.U = zonotope(0, 0.05);
params.paramInt = interval(0.4, 0.6);
params.tStart = 0;
params.tFinal = 0.1;
% Note: uTrans is set by validateOptions, not needed here

% Options
options.timeStep = 0.1;
options.taylorTerms = 4;
options.intermediateTerms = 2;
options.zonotopeOrder = 10;
options.reductionTechnique = 'girard';
options.alg = 'lin';
options.tensorOrder = 2;
options.maxError = [1; 1];

% Call linearize to see what linParamSys is created
fprintf('Calling linearize...\n');
try
    [obj, linsys, linParams, linOptions] = linearize(sys, params.R0, params, options);
    
    fprintf('SUCCESS! Linearized system created.\n');
    fprintf('\n');
    fprintf('Linearized system properties:\n');
    fprintf('  Class: %s\n', class(linsys));
    fprintf('  linsys.A type: %s\n', class(linsys.A));
    if isa(linsys.A, 'matZonotope')
        fprintf('  linsys.A.numgens: %d\n', linsys.A.numgens());
    end
    fprintf('  linsys.constParam: %d\n', linsys.constParam);
    fprintf('  linsys.B type: %s\n', class(linsys.B));
    if isnumeric(linsys.B)
        fprintf('  linsys.B value: %s\n', mat2str(linsys.B));
    end
    
    fprintf('\n');
    fprintf('This will determine which code path in priv_mappingMatrix:\n');
    fprintf('  Condition: isa(obj.A,''matZonotope'') && (obj.A.numgens() == 1) && obj.constParam\n');
    condition1 = isa(linsys.A, 'matZonotope');
    condition2 = condition1 && (linsys.A.numgens() == 1);
    condition3 = condition2 && linsys.constParam;
    fprintf('  isa(obj.A,''matZonotope''): %d\n', condition1);
    if condition1
        fprintf('  obj.A.numgens() == 1: %d\n', linsys.A.numgens() == 1);
    end
    fprintf('  obj.constParam: %d\n', linsys.constParam);
    fprintf('  All conditions true: %d\n', condition3);
    
    if condition3
        fprintf('\n  -> Will call expmOneParam (which sets iPow = [])\n');
        fprintf('  -> This is likely the problem!\n');
    else
        fprintf('\n  -> Will call expmMixed or expmIndMixed\n');
        if linsys.constParam
            fprintf('  -> expmMixed (should work)\n');
        else
            fprintf('  -> expmIndMixed (should work)\n');
        end
    end
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Stack:\n');
    for i=1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
