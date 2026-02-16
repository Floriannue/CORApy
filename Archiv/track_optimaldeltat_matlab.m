% track_optimaldeltat_matlab - Track aux_optimaldeltat inputs and outputs in MATLAB

addpath('cora_matlab/models/auxiliary/jetEngine');

dim_x = 2;
params.tFinal = 8;
params.R0 = zonotope([[1;1],0.1*diag(ones(dim_x,1))]);
params.U = zonotope(0);

options.alg = 'lin-adaptive';
options.trackOptimaldeltat = true;

% Initialize global log
global optimaldeltatLogGlobal;
optimaldeltatLogGlobal = [];

sys = nonlinearSys(@jetEngine,dim_x,1);

fprintf('Running MATLAB with aux_optimaldeltat tracking...\n');
[R,~,opt] = reach(sys,params,options);

% Extract log from global variable
if ~isempty(optimaldeltatLogGlobal)
    log = optimaldeltatLogGlobal;
    fprintf('\nCaptured %d aux_optimaldeltat calls\n', length(log));
    
    % Save to file
    save('optimaldeltat_matlab_log.mat', 'log');
    fprintf('Saved to optimaldeltat_matlab_log.mat\n');
    
    % Show first 10 entries
    fprintf('\nFirst 10 entries:\n');
    for i = 1:min(10, length(log))
        fprintf('\nStep %d:\n', log(i).step);
        fprintf('  deltat (finitehorizon): %.6e\n', log(i).deltat);
        fprintf('  varphimin: %.6f\n', log(i).varphimin);
        fprintf('  zetaP: %.6f\n', log(i).zetaP);
        fprintf('  rR: %.6e\n', log(i).rR);
        fprintf('  rerr1: %.6e\n', log(i).rerr1);
        fprintf('  varphiprod (first 5): ');
        fprintf('%.6e ', log(i).varphiprod(1:min(5, length(log(i).varphiprod))));
        fprintf('\n');
        fprintf('  deltats (first 5): ');
        fprintf('%.6e ', log(i).deltats(1:min(5, length(log(i).deltats))));
        fprintf('\n');
        fprintf('  objfuncset (first 5): ');
        fprintf('%.6e ', log(i).objfuncset(1:min(5, length(log(i).objfuncset))));
        fprintf('\n');
        fprintf('  bestIdxnew: %d\n', log(i).bestIdxnew);
        fprintf('  deltatest (selected): %.6e\n', log(i).deltatest);
        fprintf('  kprimeest: %.6f\n', log(i).kprimeest);
    end
else
    fprintf('No _optimaldeltat_log found\n');
end
