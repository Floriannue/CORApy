% Detailed investigation of eAt (exponential matrix) computation in MATLAB
% MATLAB version

% Setup
dim_x = 6;
params.R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2 * eye(dim_x));
params.U = zonotope(zeros(1, 1), 0.005 * eye(1));
params.tFinal = 4;
params.uTrans = zeros(1, 1);

options.timeStep = 4;
options.taylorTerms = 4;
options.zonotopeOrder = 50;
options.alg = 'lin';
options.tensorOrder = 2;

% System
tank = nonlinearSys(@tank6Eq, 6, 1);

% Options check
[params,options] = validateOptions(tank,params,options,'FunctionName','reach');

% Compute derivatives
derivatives(tank, options);

% Linearize
[sys, linsys, linParams, linOptions] = linearize(tank, params.R0, params, options);

A = linsys.A;
timeStep = options.timeStep;
taylorTerms = options.taylorTerms;

fprintf('=== DETAILED eAt INVESTIGATION (MATLAB) ===\n\n');
fprintf('System matrix A:\n');
disp(A);
fprintf('\nTime step: %.15f\n', timeStep);
fprintf('Taylor terms: %d\n', taylorTerms);

% Method 1: Direct expm computation (what MATLAB uses)
fprintf('\n=== Method 1: MATLAB expm ===\n');
eAt_expm = expm(A * timeStep);
fprintf('eAt from expm:\n');
disp(eAt_expm);
fprintf('\neAt[1, :]: [%s]\n', num2str(eAt_expm(1, :), '%.15f '));
fprintf('eAt diagonal: [%s]\n', num2str(diag(eAt_expm), '%.15f '));

% Method 2: Manual Taylor series expansion
fprintf('\n=== Method 2: Manual Taylor Series ===\n');
I = eye(size(A));
eAt_taylor = I;
A_power = I;

fprintf('Taylor series terms:\n');
for i = 1:taylorTerms
    A_power = A_power * A;
    term = A_power * (timeStep^i) / factorial(i);
    eAt_taylor = eAt_taylor + term;
    fprintf('  Term %d: factor = %.10f\n', i, timeStep^i / factorial(i));
    fprintf('    A^%d * dt^%d/%d! (first row): [%s]\n', i, i, i, num2str(term(1, :), '%.15f '));
end

fprintf('\neAt from Taylor (first %d terms):\n', taylorTerms);
disp(eAt_taylor);
fprintf('eAt_taylor[1, :]: [%s]\n', num2str(eAt_taylor(1, :), '%.15f '));
fprintf('eAt_taylor diagonal: [%s]\n', num2str(diag(eAt_taylor), '%.15f '));

% Method 3: Extended Taylor series (more terms)
fprintf('\n=== Method 3: Extended Taylor Series (20 terms) ===\n');
eAt_taylor_ext = I;
A_power_ext = I;
for i = 1:20
    A_power_ext = A_power_ext * A;
    term = A_power_ext * (timeStep^i) / factorial(i);
    eAt_taylor_ext = eAt_taylor_ext + term;
end
fprintf('eAt_taylor_ext[1, :]: [%s]\n', num2str(eAt_taylor_ext(1, :), '%.15f '));
fprintf('eAt_taylor_ext diagonal: [%s]\n', num2str(diag(eAt_taylor_ext), '%.15f '));

% Compare differences
fprintf('\n=== Comparison ===\n');
diff_expm_taylor = eAt_expm - eAt_taylor;
diff_expm_taylor_ext = eAt_expm - eAt_taylor_ext;
fprintf('Difference (expm - Taylor %d terms)[1, :]: [%s]\n', taylorTerms, num2str(diff_expm_taylor(1, :), '%.15e '));
fprintf('Difference (expm - Taylor 20 terms)[1, :]: [%s]\n', num2str(diff_expm_taylor_ext(1, :), '%.15e '));
fprintf('Max abs diff (expm - Taylor %d): %.15e\n', taylorTerms, max(abs(diff_expm_taylor(:))));
fprintf('Max abs diff (expm - Taylor 20): %.15e\n', max(abs(diff_expm_taylor_ext(:))));

% Check what getTaylor returns
fprintf('\n=== Method 4: getTaylor (from taylorLinSys) ===\n');
if isprop(linsys, 'taylor') && ~isempty(linsys.taylor)
    eAt_getTaylor = getTaylor(linsys.taylor, 'eAdt', struct('timeStep', timeStep));
    fprintf('eAt from getTaylor:\n');
    disp(eAt_getTaylor);
    fprintf('eAt_getTaylor[1, :]: [%s]\n', num2str(eAt_getTaylor(1, :), '%.15f '));
    fprintf('eAt_getTaylor diagonal: [%s]\n', num2str(diag(eAt_getTaylor), '%.15f '));
    
    diff_expm_getTaylor = eAt_expm - eAt_getTaylor;
    fprintf('\nDifference (expm - getTaylor)[1, :]: [%s]\n', num2str(diff_expm_getTaylor(1, :), '%.15e '));
    fprintf('Max abs diff (expm - getTaylor): %.15e\n', max(abs(diff_expm_getTaylor(:))));
    fprintf('Are expm and getTaylor identical? %d\n', isequal(eAt_expm, eAt_getTaylor, 1e-15));
end

% Check matrix properties
fprintf('\n=== Matrix Properties ===\n');
fprintf('A norm (Frobenius): %.15f\n', norm(A, 'fro'));
fprintf('A norm (inf): %.15f\n', norm(A, inf));
fprintf('A*timeStep norm (Frobenius): %.15f\n', norm(A * timeStep, 'fro'));
fprintf('A*timeStep norm (inf): %.15f\n', norm(A * timeStep, inf));

% Check eigenvalues
eigenvals = eig(A);
fprintf('\nA eigenvalues: [%s]\n', num2str(eigenvals', '%.15f '));
fprintf('A*timeStep eigenvalues: [%s]\n', num2str(eigenvals * timeStep', '%.15f '));
eAt_eigenvals = eig(eAt_expm);
fprintf('exp(A*timeStep) eigenvalues (should be exp of above): [%s]\n', num2str(eAt_eigenvals', '%.15f '));
fprintf('exp(A*timeStep eigenvalues) directly: [%s]\n', num2str(exp(eigenvals * timeStep)', '%.15f '));
