% test_jetEngine_matlab - Run jetEngine adaptive reachability in MATLAB
% This matches test_nonlinearSys_reach_adaptive_01_jetEngine.py

% Add path to jetEngine hessian functions
addpath('cora_matlab/models/auxiliary/jetEngine');

% system dimension
dim_x = 2;

% parameters (EXACTLY matching original MATLAB example_nonlinear_reach_12_adaptive.m)
params.tFinal = 8;
params.R0 = zonotope([[1;1],0.1*diag(ones(dim_x,1))]);
params.U = zonotope(0);
% Note: params.tStart is not set in original example (defaults to 0)

% algorithm parameters (EXACTLY matching original MATLAB example)
% Original example only sets: options.alg = 'lin-adaptive';
% No progress, progressInterval, or verbose options
options.alg = 'lin-adaptive';

% init system
sys = nonlinearSys(@jetEngine,dim_x,1);

% run reachability analysis
fprintf('Starting MATLAB reachability analysis...\n');
adapTime = tic;
[R,~,opt] = reach(sys,params,options);
tComp = toc(adapTime);

fprintf('MATLAB computation completed in %.2f seconds\n', tComp);

% Extract results
endset = R.timePoint.set{end};
gamma_o = 2*rad(interval(endset));

fprintf('Final set radius: %.6e\n', max(gamma_o));
fprintf('Number of time points: %d\n', length(R.timePoint.set));
fprintf('Final time: %.6f\n', R.timePoint.time{end});

% Save key results for comparison
results.tComp = tComp;
results.numSteps = length(R.timePoint.set);
results.finalTime = R.timePoint.time{end};
results.finalRadius = max(gamma_o);
results.options_alg = opt.alg;  % Should be 'lin' after 'adaptive' removal

save('jetEngine_matlab_results.mat', 'results', 'R', 'opt');

fprintf('\nResults saved to jetEngine_matlab_results.mat\n');
fprintf('Key results:\n');
fprintf('  Computation time: %.2f seconds\n', results.tComp);
fprintf('  Number of steps: %d\n', results.numSteps);
fprintf('  Final time: %.6f\n', results.finalTime);
fprintf('  Final radius: %.6e\n', results.finalRadius);
fprintf('  Final alg: %s\n', results.options_alg);
