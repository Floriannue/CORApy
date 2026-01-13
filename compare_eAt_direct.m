% Direct comparison of eAt computation - no system setup required
% Just compute eAt from the A matrix directly

% A matrix from Python output
A = [-0.0234201,   0,          0,          0,          0,         -0.01;
     0.0234201,  -0.01677445,  0,          0,          0,          0;
     0,          0.01677445, -0.01661043,  0,          0,          0;
     0,          0,          0.01661043, -0.02304648,  0,          0;
     0,          0,          0,          0.02304648, -0.01062954,  0;
     0,          0,          0,          0,          0.01062954, -0.01629874];

timeStep = 4;

fprintf('=== MATLAB eAt COMPUTATION ===\n\n');
fprintf('A matrix:\n');
disp(A);
fprintf('Time step: %.15f\n\n', timeStep);

% Method 1: Direct expm
fprintf('=== Method 1: MATLAB expm ===\n');
eAt_expm = expm(A * timeStep);
fprintf('eAt from expm:\n');
disp(eAt_expm);
fprintf('eAt[1, :] (first row, MATLAB indexing): [%s]\n', num2str(eAt_expm(1, :), '%.15e '));
fprintf('eAt diagonal: [%s]\n', num2str(diag(eAt_expm), '%.15e '));

% Method 2: Manual Taylor series (4 terms)
fprintf('\n=== Method 2: Manual Taylor Series (4 terms) ===\n');
I = eye(size(A));
eAt_taylor = I;
A_power = I;
taylorTerms = 4;

for i = 1:taylorTerms
    A_power = A_power * A;
    term = A_power * (timeStep^i) / factorial(i);
    eAt_taylor = eAt_taylor + term;
end

fprintf('eAt from Taylor (4 terms)[1, :]: [%s]\n', num2str(eAt_taylor(1, :), '%.15e '));
fprintf('eAt from Taylor (4 terms) diagonal: [%s]\n', num2str(diag(eAt_taylor), '%.15e '));

diff_taylor = eAt_expm - eAt_taylor;
fprintf('Difference (expm - Taylor 4 terms) max abs: %.15e\n', max(abs(diff_taylor(:))));

% Method 3: Extended Taylor series (20 terms)
fprintf('\n=== Method 3: Extended Taylor Series (20 terms) ===\n');
eAt_taylor_ext = I;
A_power_ext = I;
for i = 1:20
    A_power_ext = A_power_ext * A;
    term = A_power_ext * (timeStep^i) / factorial(i);
    eAt_taylor_ext = eAt_taylor_ext + term;
end

fprintf('eAt from Taylor (20 terms)[1, :]: [%s]\n', num2str(eAt_taylor_ext(1, :), '%.15e '));
fprintf('eAt from Taylor (20 terms) diagonal: [%s]\n', num2str(diag(eAt_taylor_ext), '%.15e '));

diff_taylor_ext = eAt_expm - eAt_taylor_ext;
fprintf('Difference (expm - Taylor 20 terms) max abs: %.15e\n', max(abs(diff_taylor_ext(:))));

% Matrix properties
fprintf('\n=== Matrix Properties ===\n');
fprintf('A norm (Frobenius): %.15f\n', norm(A, 'fro'));
fprintf('A norm (inf): %.15f\n', norm(A, inf));
fprintf('A*timeStep norm (Frobenius): %.15f\n', norm(A * timeStep, 'fro'));
fprintf('A*timeStep norm (inf): %.15f\n', norm(A * timeStep, inf));

% Eigenvalues
eigenvals = eig(A);
fprintf('\nA eigenvalues: [%s]\n', num2str(eigenvals', '%.15e '));
fprintf('A*timeStep eigenvalues: [%s]\n', num2str(eigenvals * timeStep', '%.15e '));
eAt_eigenvals = eig(eAt_expm);
fprintf('exp(A*timeStep) eigenvalues: [%s]\n', num2str(eAt_eigenvals', '%.15e '));
fprintf('exp(A*timeStep eigenvalues) directly: [%s]\n', num2str(exp(eigenvals * timeStep)', '%.15e '));

% Output for Python comparison
fprintf('\n=== VALUES FOR PYTHON COMPARISON ===\n');
fprintf('Python should match these values:\n');
fprintf('eAt[0, :] (Python indexing, MATLAB row 1): [%s]\n', num2str(eAt_expm(1, :), '%.15e '));
fprintf('eAt diagonal: [%s]\n', num2str(diag(eAt_expm), '%.15e '));
