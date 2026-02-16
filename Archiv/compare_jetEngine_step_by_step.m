% compare_jetEngine_step_by_step - Track intermediate values in MATLAB
% This will help identify where Python and MATLAB diverge

addpath('cora_matlab/models/auxiliary/jetEngine');

dim_x = 2;
params.tFinal = 8;
params.R0 = zonotope([[1;1],0.1*diag(ones(dim_x,1))]);
params.U = zonotope(0);

options.alg = 'lin-adaptive';
options.traceIntermediateValues = true;  % Enable tracking

sys = nonlinearSys(@jetEngine,dim_x,1);

fprintf('Running MATLAB with intermediate value tracking...\n');
[R,~,opt] = reach(sys,params,options);

fprintf('\nMATLAB Results:\n');
fprintf('  Number of steps: %d\n', length(R.timePoint.set));
fprintf('  Final time: %.10f\n', R.timePoint.time{end});

% Extract time steps
tVec = query(R,'tVec');
fprintf('  Time step stats:\n');
fprintf('    Min: %.6e\n', min(tVec));
fprintf('    Max: %.6e\n', max(tVec));
fprintf('    Mean: %.6e\n', mean(tVec));
fprintf('    Last 10 sum: %.6e\n', sum(tVec(end-9:end)));

% Check abortion condition at final step
N = 10;
k = length(tVec);
lastNsteps = sum(tVec(end-min(N,k)+1:end));
remTime = params.tFinal - R.timePoint.time{end};
if lastNsteps > 0
    ratio = remTime / lastNsteps;
    fprintf('  Abortion check at final step:\n');
    fprintf('    remTime: %.6f\n', remTime);
    fprintf('    lastNsteps: %.6e\n', lastNsteps);
    fprintf('    ratio: %.2e\n', ratio);
    fprintf('    Would abort: %d\n', ratio > 1e9);
end
