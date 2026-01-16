% Debug script to check dimensions in priv_reach_wrappingfree
% Compare against Python implementation

clear; close all; clc;

% Create a simple 2D system
A = [-0.1, -2; 2, -0.1];
B = [1; 0];
sys = linearSys(A, B);

% Parameters
params.tStart = 0;
params.tFinal = 0.2;
params.R0 = zonotope([1; 1], 0.1*eye(2));
params.U = zonotope(0, 0.01);
params.uTrans = 0.5;

% Options
options.timeStep = 0.05;
options.taylorTerms = 4;
options.zonotopeOrder = 20;
options.linAlg = 'wrapping-free';

% Put system into canonical form
[sys_canon, U, u, V, v] = sys.canonicalForm(params.U, params.uTrans, ...
    params.W, params.V, zeros(sys.nrOfInputs, 1));

% Compute reachable sets for first step
[Rtp, Rti, Htp, Hti, PU, Pu, ~, C_input] = sys_canon.oneStep(...
    params.R0, U, u, options.timeStep, options.taylorTerms);

fprintf('=== After oneStep ===\n');
fprintf('PU type: %s\n', class(PU));
if isa(PU, 'zonotope')
    fprintf('PU center shape: %s\n', mat2str(size(PU.c)));
    fprintf('PU generators shape: %s\n', mat2str(size(PU.G)));
end

% Read out propagation matrix
eAdt = getTaylor(sys_canon, 'eAdt', struct('timeStep', options.timeStep));
fprintf('\neAdt shape: %s\n', mat2str(size(eAdt)));

% Save particular solution
PU_next = PU;
fprintf('\nPU_next type: %s\n', class(PU_next));
if isa(PU_next, 'zonotope')
    fprintf('PU_next center shape: %s\n', mat2str(size(PU_next.c)));
end

% Convert PU to interval
PU = interval(PU);
fprintf('\nPU (after interval conversion) type: %s\n', class(PU));
fprintf('PU.inf shape: %s\n', mat2str(size(PU.inf)));
fprintf('PU.sup shape: %s\n', mat2str(size(PU.sup)));

% Check Pu
if isa(Pu, 'zonotope')
    Pu_c = center(Pu);
    fprintf('\nPu type: %s\n', class(Pu));
    fprintf('Pu center shape: %s\n', mat2str(size(Pu_c)));
    Pu_int = interval(Pu) - Pu_c;
    fprintf('Pu_int type: %s\n', class(Pu_int));
    fprintf('Pu_int.inf shape: %s\n', mat2str(size(Pu_int.inf)));
    fprintf('Pu_int.sup shape: %s\n', mat2str(size(Pu_int.sup)));
else
    Pu_c = Pu;
    Pu_int = zeros(sys.nrOfDims, 1);
    fprintf('\nPu is numeric, shape: %s\n', mat2str(size(Pu)));
    fprintf('Pu_int shape: %s\n', mat2str(size(Pu_int)));
end

% Check C_input
fprintf('\nC_input type: %s\n', class(C_input));
if isa(C_input, 'zonotope')
    fprintf('C_input center shape: %s\n', mat2str(size(C_input.c)));
    fprintf('C_input generators shape: %s\n', mat2str(size(C_input.G)));
end

% Simulate one iteration
fprintf('\n=== After one iteration ===\n');

% Propagate particular solution
PU_next = eAdt * PU_next;
fprintf('PU_next (after eAdt multiplication) type: %s\n', class(PU_next));
if isa(PU_next, 'zonotope')
    fprintf('PU_next center shape: %s\n', mat2str(size(PU_next.c)));
end

% Convert to interval
PU_next_interval = interval(PU_next);
fprintf('PU_next_interval type: %s\n', class(PU_next_interval));
fprintf('PU_next_interval.inf shape: %s\n', mat2str(size(PU_next_interval.inf)));
fprintf('PU_next_interval.sup shape: %s\n', mat2str(size(PU_next_interval.sup)));

% Add intervals
PU_temp = PU + PU_next_interval;
fprintf('\nPU_temp (PU + PU_next_interval) type: %s\n', class(PU_temp));
fprintf('PU_temp.inf shape: %s\n', mat2str(size(PU_temp.inf)));
fprintf('PU_temp.sup shape: %s\n', mat2str(size(PU_temp.sup)));

PU = PU + PU_next_interval + Pu_int;
fprintf('\nPU (final) type: %s\n', class(PU));
fprintf('PU.inf shape: %s\n', mat2str(size(PU.inf)));
fprintf('PU.sup shape: %s\n', mat2str(size(PU.sup)));

% Check Hti
fprintf('\nHti type: %s\n', class(Hti));
if isa(Hti, 'zonotope')
    fprintf('Hti center shape: %s\n', mat2str(size(Hti.c)));
    fprintf('Hti generators shape: %s\n', mat2str(size(Hti.G)));
end

% Try to add Hti + PU
fprintf('\n=== Attempting Hti + PU ===\n');
fprintf('Hti dimension: %d\n', dim(Hti));
fprintf('PU dimension: %d\n', dim(PU));

try
    Rti_test = Hti + PU;
    fprintf('SUCCESS: Hti + PU works\n');
    fprintf('Rti_test type: %s\n', class(Rti_test));
    if isa(Rti_test, 'zonotope')
        fprintf('Rti_test center shape: %s\n', mat2str(size(Rti_test.c)));
    end
catch ME
    fprintf('ERROR: %s\n', ME.message);
end
