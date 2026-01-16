% Simplified debug script for deleteRedundantSets
% Tests only the cases that work without P field issues

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
fprintf(fid, 'Rold has internalCount: %d\n', isfield(Rold, 'internalCount'));
fprintf(fid, 'options.reductionInterval = %d\n', options.reductionInterval);

% Execute
R_result = deleteRedundantSets(R, Rold, options);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'R_result.internalCount = %d\n', R_result.internalCount);
fprintf(fid, 'R_result.tp length: %d\n', length(R_result.tp));
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
end
fprintf(fid, '\n');

fprintf(fid, 'Note: Test 2 (internalCount=1->2) and Test 4 (intersection) require Rold.P field.\n');
fprintf(fid, 'These are skipped as they require proper setup from previous reductionInterval step.\n');

fclose(fid);
fprintf('MATLAB deleteRedundantSets tests completed. Results saved to matlab_deleteRedundantSets_output.txt\n');
