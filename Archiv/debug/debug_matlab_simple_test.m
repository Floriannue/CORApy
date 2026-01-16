% Simple test to reproduce the issue
% Add CORA to path if needed
if ~exist('zonotope', 'file')
    addpath(genpath('cora_matlab'));
end

fprintf('=== Simple Test ===\n\n');

% Create system exactly as in the working RLC test
[matZ_A, matZ_B] = RLCcircuit();
sys = linParamSys(matZ_A, eye(dim(matZ_A,1)));

% Simple parameters matching the structure
n = dim(matZ_A,1);
Rinit = zonotope(zeros(n,1), 0.1*eye(n));

% Parameters - based on linearize.m analysis
% After linearization with B=1, Uconst and uTrans are in state space
params.Uconst = zonotope(zeros(n,1), 0.05*eye(n));
params.U = zonotope(zeros(n,1), 0.05*eye(n));
params.uTrans = 0.1*ones(n,1);

% Options
options.timeStep = 0.001;
options.taylorTerms = 8;
options.intermediateTerms = 2;
options.reductionTechnique = 'girard';
options.zonotopeOrder = 400;
options.compTimePoint = true;
options.originContained = false;

fprintf('System: %dx%d, %d generators\n', n, n, matZ_A.numgens);
fprintf('Options: intermediateTerms=%d, taylorTerms=%d\n', options.intermediateTerms, options.taylorTerms);
fprintf('Loop in priv_highOrderMappingMatrix: i = %d:%d\n', options.intermediateTerms+1, options.taylorTerms);
fprintf('\n');

% Try calling initReach_inputDependence
fprintf('Calling initReach_inputDependence...\n');
try
    [sys_out, Rfirst, options_out] = initReach_inputDependence(sys, Rinit, params, options);
    fprintf('  SUCCESS!\n');
    fprintf('  Rfirst.tp center shape: %s\n', mat2str(size(center(Rfirst.tp))));
    fprintf('  Rfirst.ti center shape: %s\n', mat2str(size(center(Rfirst.ti))));
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    fprintf('  Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    if length(ME.stack) > 1
        fprintf('  Called from: %s (line %d)\n', ME.stack(2).name, ME.stack(2).line);
    end
end
