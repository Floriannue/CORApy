% Debug script to verify deleteRedundantSets test against MATLAB
% This generates exact input/output pairs for Python tests
%
% Based on deleteRedundantSets.m

% Add CORA to path
addpath(genpath('cora_matlab'));

fid = fopen('matlab_deleteRedundantSets_output.txt', 'w');
fprintf(fid, 'MATLAB deleteRedundantSets Test Output\n');
fprintf(fid, '======================================\n\n');

%% Test 1: First run (no Rold.internalCount)
fprintf(fid, 'Test 1: First run (no Rold.internalCount)\n');
fprintf(fid, '----------------------------------------\n');

R = struct();
R.tp = {struct('set', zonotope([0; 0], 0.1*eye(2)), 'error', [0.01; 0.01])};
R.ti = {zonotope([0; 0], 0.1*eye(2))};
Rold = struct();  % Empty - no internalCount
options = struct();
options.reductionInterval = 3;
options.maxError = [0.1; 0.1];

fprintf(fid, 'Input:\n');
fprintf(fid, 'R.tp length: %d\n', length(R.tp));
fprintf(fid, 'R.ti length: %d\n', length(R.ti));
fprintf(fid, 'Rold has internalCount: %d\n', isfield(Rold, 'internalCount'));
fprintf(fid, 'options.reductionInterval = %d\n', options.reductionInterval);

% Execute
R_result = deleteRedundantSets(R, Rold, options);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'R_result.internalCount = %d\n', R_result.internalCount);
fprintf(fid, 'R_result.tp length: %d\n', length(R_result.tp));
fprintf(fid, '\n');

%% Test 2: Increment count (not reaching reductionInterval or 2)
fprintf(fid, 'Test 2: Increment count (internalCount = 1 -> 2, but no P field)\n');
fprintf(fid, '------------------------------------------------------------------\n');
fprintf(fid, 'Note: This test case would fail because internalCount=2 requires Rold.P\n');
fprintf(fid, 'Skipping execution - see Test 4 for internalCount=2 with P field\n');
fprintf(fid, '\n');

%% Test 3: Reduction interval (internalCount == reductionInterval)
fprintf(fid, 'Test 3: Reduction interval (internalCount == reductionInterval)\n');
fprintf(fid, '---------------------------------------------------------------\n');

R3 = struct();
R3.tp = {
    struct('set', zonotope([0; 0], 0.1*eye(2)), 'error', [0.01; 0.01]),
    struct('set', zonotope([1; 1], 0.1*eye(2)), 'error', [0.01; 0.01])
};
R3.ti = {
    zonotope([0; 0], 0.1*eye(2)),
    zonotope([1; 1], 0.1*eye(2))
};
Rold3 = struct();
Rold3.internalCount = 2;  % Will become 3, then reset to 1
options3 = struct();
options3.reductionInterval = 3;
options3.maxError = [0.1; 0.1];

fprintf(fid, 'Input:\n');
fprintf(fid, 'Rold3.internalCount = %d\n', Rold3.internalCount);
fprintf(fid, 'options3.reductionInterval = %d\n', options3.reductionInterval);
fprintf(fid, 'R3.tp length: %d\n', length(R3.tp));

% Execute
R_result3 = deleteRedundantSets(R3, Rold3, options3);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'R_result3.internalCount = %d\n', R_result3.internalCount);
fprintf(fid, 'R_result3 has P field: %d\n', isfield(R_result3, 'P'));
if isfield(R_result3, 'P')
    fprintf(fid, 'R_result3.P length: %d\n', length(R_result3.P));
    if length(R_result3.P) > 0
        fprintf(fid, 'R_result3.P{1} type: %s\n', class(R_result3.P{1}));
    end
end
fprintf(fid, '\n');

%% Test 4: Set intersection (internalCount == 2)
fprintf(fid, 'Test 4: Set intersection (internalCount == 2)\n');
fprintf(fid, '----------------------------------------------\n');

R4 = struct();
R4.tp = {
    struct('set', zonotope([0; 0], 0.1*eye(2)), 'error', [0.01; 0.01]),
    struct('set', zonotope([0.5; 0.5], 0.1*eye(2)), 'error', [0.01; 0.01])
};
R4.ti = {
    zonotope([0; 0], 0.1*eye(2)),
    zonotope([0.5; 0.5], 0.1*eye(2))
};
Rold4 = struct();
Rold4.internalCount = 1;  % Will become 2
Rold4.P = {polytope(zonotope([0.2; 0.2], 0.15*eye(2)))};  % Previous polytope
options4 = struct();
options4.reductionInterval = 3;
options4.maxError = [0.1; 0.1];

fprintf(fid, 'Input:\n');
fprintf(fid, 'Rold4.internalCount = %d\n', Rold4.internalCount);
fprintf(fid, 'Rold4.P length: %d\n', length(Rold4.P));
fprintf(fid, 'R4.tp length: %d\n', length(R4.tp));

% Execute
R_result4 = deleteRedundantSets(R4, Rold4, options4);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'R_result4.internalCount = %d\n', R_result4.internalCount);
fprintf(fid, 'R_result4.tp length: %d\n', length(R_result4.tp));
fprintf(fid, '\n');

fclose(fid);
fprintf('MATLAB deleteRedundantSets tests completed. Results saved to matlab_deleteRedundantSets_output.txt\n');
