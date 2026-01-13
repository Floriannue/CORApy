% Debug script to test initReach_inputDependence via proper path
% This uses nonlinParamSys -> reach -> linReach -> linearize -> initReach_inputDependence
% Add CORA to path if needed
if ~exist('zonotope', 'file')
    addpath(genpath('cora_matlab'));
end

fprintf('=== Test initReach_inputDependence via proper path (nonlinParamSys) ===\n\n');

% Create a simple 2D nonlinear parametric system
% System: x' = [x(2); -x(1) - p(1)*x(2) + u]
% This is a simple damped oscillator with uncertain damping parameter
f = @(x, u, p) [x(2); -x(1) - p(1)*x(2) + u(1)];

% Create nonlinParamSys (no output function needed for reachability)
sys = nonlinParamSys('simple_oscillator', f);

% Parameters
params.R0 = zonotope([0; 0], [0.1, 0; 0, 0.1]);
params.U = zonotope(0, 0.05);  % Input dimension (1D)
params.uTrans = 0.1;  % Input dimension (1D)
params.paramInt = interval(0.4, 0.6);  % Uncertain parameter (damping)
params.tStart = 0;
params.tFinal = 0.1;  % Short time horizon for testing

% Options
options.timeStep = 0.1;
options.taylorTerms = 4;
options.intermediateTerms = 2;
options.zonotopeOrder = 10;
options.reductionTechnique = 'girard';
options.alg = 'lin';  % Linear algorithm
options.tensorOrder = 2;
options.maxError = [1; 1];  % State dimension
options.compOutputSet = false;  % Disable output set computation
% Note: compTimePoint is set automatically, don't set it manually

fprintf('System setup:\n');
fprintf('  System: 2D nonlinear parametric system\n');
fprintf('  paramInt: interval(%.15g, %.15g)\n', infimum(params.paramInt), supremum(params.paramInt));
fprintf('  U: zonotope in input dimension (1D)\n');
fprintf('  uTrans: %.15g (input dimension)\n', params.uTrans);
fprintf('  timeStep: %.15g\n', options.timeStep);
fprintf('  taylorTerms: %d\n', options.taylorTerms);
fprintf('  intermediateTerms: %d\n', options.intermediateTerms);
fprintf('\n');

% Call reach - this will trigger linReach -> linearize -> initReach_inputDependence
fprintf('Calling reach() on nonlinParamSys...\n');
fprintf('  This will call: reach -> linReach -> linearize -> initReach_inputDependence\n');
fprintf('\n');

try
    R = reach(sys, params, options);
    
    fprintf('SUCCESS! Reachability analysis completed.\n');
    fprintf('\n');
    fprintf('Results:\n');
    fprintf('  Number of time steps: %d\n', length(R.timeInterval.set));
    if ~isempty(R.timeInterval.set)
        fprintf('  Final time interval set center: [%.15g; %.15g]\n', center(R.timeInterval.set{end}));
    end
    if ~isempty(R.timePoint.set)
        fprintf('  Final time point set center: [%.15g; %.15g]\n', center(R.timePoint.set{end}));
    end
    
    % Extract the first reachable set (which uses initReach_inputDependence)
    fprintf('\n');
    fprintf('First reachable set (from initReach_inputDependence):\n');
    if ~isempty(R.timeInterval.set) && length(R.timeInterval.set) >= 1
        Rfirst_ti = R.timeInterval.set{1};
        fprintf('  Rfirst.ti center: [%.15g; %.15g]\n', center(Rfirst_ti));
        fprintf('  Rfirst.ti generators shape: %s\n', mat2str(size(generators(Rfirst_ti))));
    end
    if ~isempty(R.timePoint.set) && length(R.timePoint.set) >= 2
        Rfirst_tp = R.timePoint.set{2};  % First time point after initial
        fprintf('  Rfirst.tp center: [%.15g; %.15g]\n', center(Rfirst_tp));
        fprintf('  Rfirst.tp generators shape: %s\n', mat2str(size(generators(Rfirst_tp))));
    end
    
    % Save detailed output for Python test
    fprintf('\n');
    fprintf('=== MATLAB I/O Pairs for Python Test ===\n');
    fprintf('\n');
    fprintf('Input parameters:\n');
    fprintf('  params.R0 center: [%.15g; %.15g]\n', center(params.R0));
    fprintf('  params.U center: %.15g\n', center(params.U));
    fprintf('  params.uTrans: %.15g\n', params.uTrans);
    fprintf('  params.paramInt: interval(%.15g, %.15g)\n', infimum(params.paramInt), supremum(params.paramInt));
    fprintf('\n');
    fprintf('Output (first reachable set):\n');
    if ~isempty(R.timeInterval.set) && length(R.timeInterval.set) >= 1
        Rfirst_ti = R.timeInterval.set{1};
        fprintf('  Rfirst.ti center: [%.15g; %.15g]\n', center(Rfirst_ti));
        Rfirst_ti_gens = generators(Rfirst_ti);
        fprintf('  Rfirst.ti generators:\n');
        for i=1:size(Rfirst_ti_gens, 2)
            fprintf('    [%.15g; %.15g]\n', Rfirst_ti_gens(1,i), Rfirst_ti_gens(2,i));
        end
    end
    if ~isempty(R.timePoint.set) && length(R.timePoint.set) >= 2
        Rfirst_tp = R.timePoint.set{2};
        fprintf('  Rfirst.tp center: [%.15g; %.15g]\n', center(Rfirst_tp));
        Rfirst_tp_gens = generators(Rfirst_tp);
        fprintf('  Rfirst.tp generators:\n');
        for i=1:size(Rfirst_tp_gens, 2)
            fprintf('    [%.15g; %.15g]\n', Rfirst_tp_gens(1,i), Rfirst_tp_gens(2,i));
        end
    end
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Stack:\n');
    for i=1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
