% Debug script to verify deleteRedundantSets test against MATLAB
% This generates exact input/output pairs for Python tests
% Uses the same structure as in post.m: Rnext = deleteRedundantSets(Rnext,R,options);
%
% Add CORA to path
addpath(genpath('cora_matlab'));

fid = fopen('matlab_deleteRedundantSets_output.txt', 'w');
fprintf(fid, 'MATLAB deleteRedundantSets Test Output (Full Setup)\n');
fprintf(fid, '==================================================\n\n');

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

%% Test 2: Increment count (internalCount = 1 -> 2, but no P field yet)
fprintf(fid, 'Test 2: Increment count (internalCount = 1 -> 2)\n');
fprintf(fid, '------------------------------------------------\n');
fprintf(fid, 'Note: When internalCount becomes 2, MATLAB code requires Rold.P to exist.\n');
fprintf(fid, 'Since P is only created at reductionInterval, this case is skipped.\n');
fprintf(fid, 'See Test 4 for the internalCount==2 case with proper P field.\n');
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
        % Print polytope info
        P1 = R_result3.P{1};
        fprintf(fid, 'R_result3.P{1}.dim: %d\n', P1.dim);
    end
end
fprintf(fid, 'R_result3.tp length: %d\n', length(R_result3.tp));
fprintf(fid, '\n');

%% Test 4: Set intersection (internalCount == 2) with Rold.P
fprintf(fid, 'Test 4: Set intersection (internalCount == 2) with Rold.P\n');
fprintf(fid, '----------------------------------------------------------\n');

% Use R_result3 from Test 3, which has P field and internalCount = 1
% This simulates what happens in a real reachability analysis:
% Step 1: reductionInterval creates P (Test 3)
% Step 2: Next call with internalCount=1 (will become 2) uses that P

% Now create R4 for the actual test (internalCount will become 2)
R4 = struct();
R4.tp = {
    struct('set', zonotope([0; 0], 0.1*eye(2)), 'error', [0.01; 0.01]),
    struct('set', zonotope([0.5; 0.5], 0.1*eye(2)), 'error', [0.01; 0.01])
};
R4.ti = {
    zonotope([0; 0], 0.1*eye(2)),
    zonotope([0.5; 0.5], 0.1*eye(2))
};
Rold4 = R_result3;  % Has P field and internalCount = 1 (will become 2)
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
if length(R_result4.tp) > 0
    fprintf(fid, 'R_result4.tp{1} has prev field: %d\n', isfield(R_result4.tp{1}, 'prev'));
    if isfield(R_result4.tp{1}, 'prev')
        fprintf(fid, 'R_result4.tp{1}.prev = %d\n', R_result4.tp{1}.prev);
    end
end
fprintf(fid, '\n');

%% Test 5: With parent field
fprintf(fid, 'Test 5: With parent field\n');
fprintf(fid, '--------------------------\n');

R5 = struct();
R5.tp = {
    struct('set', zonotope([0; 0], 0.1*eye(2)), 'error', [0.01; 0.01], 'parent', 0)
};
R5.ti = {zonotope([0; 0], 0.1*eye(2))};
Rold5 = struct();
Rold5.internalCount = 1;
options5 = struct();
options5.reductionInterval = 3;
options5.maxError = [0.1; 0.1];

fprintf(fid, 'Input:\n');
fprintf(fid, 'R5.tp{1} has parent field: %d\n', isfield(R5.tp{1}, 'parent'));
if isfield(R5.tp{1}, 'parent')
    fprintf(fid, 'R5.tp{1}.parent = %d\n', R5.tp{1}.parent);
end

% Execute
R_result5 = deleteRedundantSets(R5, Rold5, options5);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'R_result5.internalCount = %d\n', R_result5.internalCount);
fprintf(fid, 'R_result5.tp length: %d\n', length(R_result5.tp));
if length(R_result5.tp) > 0 && isfield(R_result5.tp{1}, 'parent')
    fprintf(fid, 'R_result5.tp{1}.parent = %d\n', R_result5.tp{1}.parent);
end
fprintf(fid, '\n');

fclose(fid);
fprintf('MATLAB deleteRedundantSets tests completed. Results saved to matlab_deleteRedundantSets_output.txt\n');
