% Debug script to trace intermediate values in linReach_adaptive
% This tracks all intermediate values step by step to compare with Python

addpath(genpath('cora_matlab'));

% Setup test case - use a simpler case that we can trace
params.tStart = 0;
params.tFinal = 10;
params.R0 = zonotope([0; 0], 0.1*eye(2));
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
options.maxError = 0.1 * ones(2, 1);
options.zetaphi = [0.5; 0.5];
options.zetaK = 0.1;
options.orders = ones(2, 1);
options.i = 1;
options.error_adm_horizon = 1e-10 * ones(2, 1);  % Start with small value

% Create simple 2D system
f = @(x, u) [-x(1) + x(2); -x(2) + u];
sys = nonlinearSys('simple', f, 2, 1);

fprintf('=== Tracing Intermediate Values in linReach_adaptive ===\n\n');

% Open file for detailed logging
fid = fopen('matlab_intermediate_values.txt', 'w');
fprintf(fid, '=== MATLAB Intermediate Values Trace ===\n\n');

try
    % Simulate one iteration of the inner loop
    error_adm = options.error_adm_horizon;
    fprintf('Step 0: Initial error_adm_horizon\n');
    fprintf(fid, 'Step 0: Initial error_adm_horizon\n');
    fprintf('  error_adm = [%.15e; %.15e]\n', error_adm(1), error_adm(2));
    fprintf(fid, '  error_adm = [%.15e; %.15e]\n', error_adm(1), error_adm(2));
    
    % Create Verror
    fprintf('\nStep 1: Create Verror from error_adm\n');
    fprintf(fid, '\nStep 1: Create Verror from error_adm\n');
    errG = diag(error_adm);
    Verror = zonotope(0*error_adm, errG(:, any(errG, 1)));
    Verror_center = center(Verror);
    Verror_gens = generators(Verror);
    fprintf('  Verror center: [%.15e; %.15e]\n', Verror_center(1), Verror_center(2));
    fprintf(fid, '  Verror center: [%.15e; %.15e]\n', Verror_center(1), Verror_center(2));
    fprintf('  Verror generators shape: %dx%d\n', size(Verror_gens, 1), size(Verror_gens, 2));
    fprintf(fid, '  Verror generators shape: %dx%d\n', size(Verror_gens, 1), size(Verror_gens, 2));
    fprintf('  Verror generators max abs: %.15e\n', max(max(abs(Verror_gens))));
    fprintf(fid, '  Verror generators max abs: %.15e\n', max(max(abs(Verror_gens))));
    
    % For a real trace, we'd need to call the actual functions
    % But for now, let's document what values we expect at each step
    
    fprintf('\n=== Expected Computation Flow ===\n');
    fprintf(fid, '\n=== Expected Computation Flow ===\n');
    fprintf('1. error_adm -> Verror (zonotope with diag(error_adm))\n');
    fprintf('2. Verror -> RallError (via errorSolution_adaptive)\n');
    fprintf('3. Rlinti + RallError -> Rmax\n');
    fprintf('4. Rmax -> Rred (reduced)\n');
    fprintf('5. Rred, U -> Z (cartProd)\n');
    fprintf('6. Z, H -> errorSec (0.5 * quadMap)\n');
    fprintf('7. errorSec -> VerrorDyn (after reduction)\n');
    fprintf('8. VerrorDyn -> trueError (abs(center) + sum(abs(generators)))\n');
    fprintf('9. trueError, error_adm -> perfIndCurr (max(trueError ./ error_adm))\n');
    fprintf('10. perfIndCurr -> convergence check\n');
    
    fprintf(fid, '1. error_adm -> Verror (zonotope with diag(error_adm))\n');
    fprintf(fid, '2. Verror -> RallError (via errorSolution_adaptive)\n');
    fprintf(fid, '3. Rlinti + RallError -> Rmax\n');
    fprintf(fid, '4. Rmax -> Rred (reduced)\n');
    fprintf(fid, '5. Rred, U -> Z (cartProd)\n');
    fprintf(fid, '6. Z, H -> errorSec (0.5 * quadMap)\n');
    fprintf(fid, '7. errorSec -> VerrorDyn (after reduction)\n');
    fprintf(fid, '8. VerrorDyn -> trueError (abs(center) + sum(abs(generators)))\n');
    fprintf(fid, '9. trueError, error_adm -> perfIndCurr (max(trueError ./ error_adm))\n');
    fprintf(fid, '10. perfIndCurr -> convergence check\n');
    
    fprintf('\n=== Key Values to Track ===\n');
    fprintf(fid, '\n=== Key Values to Track ===\n');
    fprintf('At each inner loop iteration:\n');
    fprintf('  - error_adm (input)\n');
    fprintf('  - RallError radius (max sum of abs generators)\n');
    fprintf('  - Rmax radius (max sum of abs generators)\n');
    fprintf('  - Z radius (max sum of abs generators)\n');
    fprintf('  - errorSec radius (max sum of abs generators)\n');
    fprintf('  - VerrorDyn radius (max sum of abs generators)\n');
    fprintf('  - trueError (vector)\n');
    fprintf('  - perfIndCurr (scalar)\n');
    fprintf('  - perfInds (array of perfIndCurr values)\n');
    
    fprintf(fid, 'At each inner loop iteration:\n');
    fprintf(fid, '  - error_adm (input)\n');
    fprintf(fid, '  - RallError radius (max sum of abs generators)\n');
    fprintf(fid, '  - Rmax radius (max sum of abs generators)\n');
    fprintf(fid, '  - Z radius (max sum of abs generators)\n');
    fprintf(fid, '  - errorSec radius (max sum of abs generators)\n');
    fprintf(fid, '  - VerrorDyn radius (max sum of abs generators)\n');
    fprintf(fid, '  - trueError (vector)\n');
    fprintf(fid, '  - perfIndCurr (scalar)\n');
    fprintf(fid, '  - perfInds (array of perfIndCurr values)\n');
    
catch ME
    fprintf('\n=== ERROR ===\n');
    fprintf('Identifier: %s\n', ME.identifier);
    fprintf('Message: %s\n', ME.message);
    fprintf(fid, '\n=== ERROR ===\n');
    fprintf(fid, 'Identifier: %s\n', ME.identifier);
    fprintf(fid, 'Message: %s\n', ME.message);
end

fclose(fid);
fprintf('\n=== Trace complete. Values saved to matlab_intermediate_values.txt ===\n');
