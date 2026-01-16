% Debug script to verify deleteRedundantSets first-run behavior
% Generates exact expected internalCount for Python test

% Create reachable set structure R with one zonotope
R.tp{1}.set = zonotope([[0;0], 0.1*eye(2)]);
R.tp{1}.error = [0.01; 0.01];
R.ti{1} = zonotope([[0;0], 0.1*eye(2)]);

% Empty Rold
Rold = struct();

% Options
options.reductionInterval = 3;
options.maxError = [0.1; 0.1];

% Run
R_out = deleteRedundantSets(R, Rold, options);

% Output
fprintf('internalCount = %d\n', R_out.internalCount);

fid = fopen('matlab_deleteRedundantSets_output.txt', 'w');
fprintf(fid, 'internalCount = %d\n', R_out.internalCount);
fclose(fid);
