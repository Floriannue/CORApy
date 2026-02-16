% Debug script to trace error_adm_horizon growth in MATLAB
% This traces the computation path step by step to understand how MATLAB handles huge values

% Load the test case
addpath(genpath('cora_matlab'));

% Setup similar to test_nonlinearSys_reach_adaptive_01_jetEngine
params.tStart = 0;
params.tFinal = 500;
params.R0 = zonotope([0; 0; 0; 0; 0], 0.1*eye(5));
params.U = zonotope(0, 0);

options = struct();
options.alg = 'lin';
options.tensorOrder = 3;
options.timeStep = 0.1;
options.taylorTerms = 10;
options.zonotopeOrder = 50;
options.reductionTechnique = 'adaptive';
options.redFactor = 0.9;
options.decrFactor = 0.5;
options.minorder = 1;
options.maxError = 0.1 * ones(5, 1);
options.zetaphi = [0.5; 0.5];
options.zetaK = 0.1;
options.orders = ones(5, 1);
options.i = 1;

% Create system
f = @(x, u) [x(2); x(3); x(4); x(5); -x(1) - x(2) - x(3) - x(4) - x(5) + u];
sys = nonlinearSys('jetEngine', f, 5, 1);

% Initialize error_adm_horizon to a huge value to simulate the problem
options.error_adm_horizon = 1e+75 * ones(5, 1);

fprintf('=== MATLAB Debug: error_adm_horizon Growth ===\n');
fprintf('Starting with error_adm_horizon = [%.6e; %.6e; %.6e; %.6e; %.6e]\n', ...
    options.error_adm_horizon(1), options.error_adm_horizon(2), ...
    options.error_adm_horizon(3), options.error_adm_horizon(4), options.error_adm_horizon(5));

% Simulate one step of linReach_adaptive
try
    error_adm = options.error_adm_horizon;
    fprintf('\n--- Step 1: Create Verror from error_adm ---\n');
    errG = diag(error_adm);
    Verror = zonotope(0*error_adm, errG(:, any(errG, 1)));
    fprintf('Verror center: [%.6e; %.6e; %.6e; %.6e; %.6e]\n', ...
        center(Verror));
    fprintf('Verror generators shape: %dx%d\n', size(generators(Verror)));
    fprintf('Verror generators max abs: %.6e\n', max(max(abs(generators(Verror)))));
    
    fprintf('\n--- Step 2: Compute RallError via errorSolution_adaptive ---\n');
    % We need a linear system for errorSolution_adaptive
    % For debugging, let's just check what happens with huge Verror
    fprintf('Verror radius estimate: %.6e\n', max(sum(abs(generators(Verror)), 2)));
    
    fprintf('\n--- Step 3: Check what happens in priv_abstractionError_adaptive ---\n');
    % Create a dummy Rmax that would result from huge RallError
    Rmax_center = [1; 1; 1; 1; 1];
    Rmax_gen = 1e+39 * eye(5);  % Simulate huge generators
    Rmax = zonotope(Rmax_center, Rmax_gen);
    fprintf('Rmax center: [%.6e; %.6e; %.6e; %.6e; %.6e]\n', Rmax_center);
    fprintf('Rmax generators max abs: %.6e\n', max(max(abs(Rmax_gen))));
    fprintf('Rmax radius estimate: %.6e\n', max(sum(abs(Rmax_gen), 2)));
    
    % Check what happens when we reduce Rmax
    fprintf('\n--- Step 4: Reduce Rmax ---\n');
    Rred = reduce(Rmax, 'adaptive', sqrt(options.redFactor));
    fprintf('Rred generators max abs: %.6e\n', max(max(abs(generators(Rred)))));
    fprintf('Rred radius estimate: %.6e\n', max(sum(abs(generators(Rred)), 2)));
    
    % Check what happens with Z = cartProd(Rred, U)
    fprintf('\n--- Step 5: Create Z = cartProd(Rred, U) ---\n');
    Z = cartProd(Rred, params.U);
    fprintf('Z generators max abs: %.6e\n', max(max(abs(generators(Z)))));
    fprintf('Z radius estimate: %.6e\n', max(sum(abs(generators(Z)), 2)));
    
    % Check what happens with quadMap
    fprintf('\n--- Step 6: Compute errorSec = 0.5 * quadMap(Z, H) ---\n');
    % We need H - for debugging, use identity
    H = cell(5, 1);
    for i = 1:5
        H{i} = eye(5);
    end
    errorSec = 0.5 * quadMap(Z, H);
    fprintf('errorSec center: [%.6e; %.6e; %.6e; %.6e; %.6e]\n', center(errorSec));
    fprintf('errorSec generators max abs: %.6e\n', max(max(abs(generators(errorSec)))));
    fprintf('errorSec radius estimate: %.6e\n', max(sum(abs(generators(errorSec)), 2)));
    
    % Check what happens with VerrorDyn
    fprintf('\n--- Step 7: Compute VerrorDyn and trueError ---\n');
    VerrorDyn = errorSec;  % Simplified: no errorLagr
    VerrorDyn = reduce(VerrorDyn, 'adaptive', 10 * options.redFactor);
    fprintf('VerrorDyn center: [%.6e; %.6e; %.6e; %.6e; %.6e]\n', center(VerrorDyn));
    fprintf('VerrorDyn generators max abs: %.6e\n', max(max(abs(generators(VerrorDyn)))));
    fprintf('VerrorDyn radius estimate: %.6e\n', max(sum(abs(generators(VerrorDyn)), 2)));
    
    trueError = abs(center(VerrorDyn)) + sum(abs(generators(VerrorDyn)), 2);
    fprintf('trueError: [%.6e; %.6e; %.6e; %.6e; %.6e]\n', ...
        trueError(1), trueError(2), trueError(3), trueError(4), trueError(5));
    fprintf('trueError max: %.6e\n', max(trueError));
    
    % Check perfIndCurr
    fprintf('\n--- Step 8: Compute perfIndCurr = max(trueError ./ error_adm) ---\n');
    perfIndCurr = max(trueError ./ error_adm);
    fprintf('perfIndCurr: %.6e\n', perfIndCurr);
    fprintf('perfIndCurr <= 1: %d\n', perfIndCurr <= 1);
    fprintf('isinf(perfIndCurr): %d\n', isinf(perfIndCurr));
    fprintf('isnan(perfIndCurr): %d\n', isnan(perfIndCurr));
    
    fprintf('\n=== MATLAB handles huge values naturally ===\n');
    fprintf('If perfIndCurr > 1, the inner loop continues and error_adm = 1.1 * trueError\n');
    fprintf('This would make error_adm even larger, causing the cycle to continue.\n');
    
catch ME
    fprintf('\n=== ERROR CAUGHT ===\n');
    fprintf('Identifier: %s\n', ME.identifier);
    fprintf('Message: %s\n', ME.message);
    fprintf('Stack:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s at line %d\n', ME.stack(i).name, ME.stack(i).line);
    end
end
