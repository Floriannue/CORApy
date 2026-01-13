% Debug script using the tank example from unit tests
% This uses nonlinParamSys -> reach -> linReach -> linearize -> initReach_inputDependence
% Add CORA to path if needed
if ~exist('zonotope', 'file')
    addpath(genpath('cora_matlab'));
end

fprintf('=== Test initReach_inputDependence via proper path (tank example) ===\n\n');

% Use the tank example from unit tests
dim_x = 6;
params.tFinal = 0.1;  % Short time for testing
params.R0 = zonotope([[2; 4; 4; 2; 10; 4], 0.2*eye(dim_x)]);
params.U = zonotope([0, 0.005]);
params.paramInt = interval(0.0148, 0.015);  % Parameter intervals for reachability analysis

% Reachability Settings
options.timeStep = 0.1;
options.taylorTerms = 4;
options.intermediateTerms = 2;  % Reduced from 4 for simpler test
options.zonotopeOrder = 10;
options.maxError = 1*ones(dim_x,1);
options.reductionInterval = 1e3;
options.tensorOrder = 2;
options.alg = 'lin';
options.compOutputSet = false;  % Disable output set computation

% System Dynamics
tankParam = nonlinParamSys(@tank6paramEq);

fprintf('System setup:\n');
fprintf('  System: 6D tank system\n');
fprintf('  paramInt: interval(%.15g, %.15g)\n', infimum(params.paramInt), supremum(params.paramInt));
fprintf('  timeStep: %.15g\n', options.timeStep);
fprintf('  taylorTerms: %d\n', options.taylorTerms);
fprintf('  intermediateTerms: %d\n', options.intermediateTerms);
fprintf('\n');

% Call reach - this will trigger linReach -> linearize -> initReach_inputDependence
fprintf('Calling reach() on nonlinParamSys...\n');
fprintf('  This will call: reach -> linReach -> linearize -> initReach_inputDependence\n');
fprintf('\n');

try
    R = reach(tankParam, params, options);
    
    fprintf('SUCCESS! Reachability analysis completed.\n');
    fprintf('\n');
    fprintf('Results:\n');
    fprintf('  Number of time steps: %d\n', length(R.timeInterval.set));
    if ~isempty(R.timeInterval.set)
        fprintf('  Final time interval set center shape: %s\n', mat2str(size(center(R.timeInterval.set{end}))));
        R_end_center = center(R.timeInterval.set{end});
        fprintf('  Final time interval set center (first 2 dims): [%.15g; %.15g]\n', ...
            R_end_center(1), R_end_center(2));
    end
    if ~isempty(R.timePoint.set) && length(R.timePoint.set) >= 2
        R_tp_end_center = center(R.timePoint.set{end});
        fprintf('  Final time point set center (first 2 dims): [%.15g; %.15g]\n', ...
            R_tp_end_center(1), R_tp_end_center(2));
    end
    
    % Extract the first reachable set (which uses initReach_inputDependence)
    fprintf('\n');
    fprintf('First reachable set (from initReach_inputDependence):\n');
    if ~isempty(R.timeInterval.set) && length(R.timeInterval.set) >= 1
        Rfirst_ti = R.timeInterval.set{1};
        Rfirst_ti_center = center(Rfirst_ti);
        fprintf('  Rfirst.ti center (first 2 dims): [%.15g; %.15g]\n', ...
            Rfirst_ti_center(1), Rfirst_ti_center(2));
        fprintf('  Rfirst.ti generators shape: %s\n', mat2str(size(generators(Rfirst_ti))));
    end
    if ~isempty(R.timePoint.set) && length(R.timePoint.set) >= 2
        Rfirst_tp = R.timePoint.set{2};  % First time point after initial
        Rfirst_tp_center = center(Rfirst_tp);
        fprintf('  Rfirst.tp center (first 2 dims): [%.15g; %.15g]\n', ...
            Rfirst_tp_center(1), Rfirst_tp_center(2));
        fprintf('  Rfirst.tp generators shape: %s\n', mat2str(size(generators(Rfirst_tp))));
    end
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Stack:\n');
    for i=1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
