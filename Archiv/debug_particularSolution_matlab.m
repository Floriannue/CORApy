% Debug script for particularSolution_timeVarying
% Matches Python test exactly

% Setup
dim_x = 6;
params.R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2*eye(dim_x));
params.U = zonotope(zeros(1,1), 0.005*eye(1));
params.tFinal = 4;
params.uTrans = zeros(1,1);

options.timeStep = 4;
options.taylorTerms = 4;
options.zonotopeOrder = 50;
options.alg = 'lin';
options.tensorOrder = 2;
options.maxError = inf(dim_x,1);

% System
tank = nonlinearSys(@tank6Eq,6,1);

% Validate options
[params,options] = validateOptions(tank,params,options,'FunctionName','reach');

% Compute derivatives (required before linearize)
derivatives(tank,options);

% Linearize
[sys, linsys, linParams, linOptions] = linearize(tank, params.R0, params, options);

% Compute particular solution
U = linParams.U;
timeStep = options.timeStep;
truncationOrder = options.taylorTerms;

PU = particularSolution_timeVarying(linsys, U, timeStep, truncationOrder);

fprintf('================================================================================\n');
fprintf('MATLAB particularSolution_timeVarying OUTPUT\n');
fprintf('================================================================================\n');

fprintf('\nPU generators shape: %dx%d\n', size(PU.G));
fprintf('\nPU generators (all columns):\n');
for i = 1:size(PU.G,2)
    fprintf('  Column %d: [%s]\n', i, sprintf('%.15e ', PU.G(:,i)));
end

% Compute delta
delta_PU = sum(abs(PU.G), 2);
fprintf('\nDelta from PU generators (sum of abs):\n');
fprintf('  [%s]\n', sprintf('%.15e ', delta_PU));

fprintf('\n================================================================================\n');
