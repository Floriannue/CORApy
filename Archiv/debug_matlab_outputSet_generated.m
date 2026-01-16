% Debug script to verify outputSet generated test cases against MATLAB
% This generates exact input/output pairs for Python tests
%
% Test cases:
% 1. No output equation (y = x)
% 2. With D matrix (direct feedthrough)
% 3. compOutputSet false
% 4. With order reduction

% Add CORA to path
addpath(genpath('cora_matlab'));

fid = fopen('matlab_outputSet_generated_output.txt', 'w');
fprintf(fid, 'MATLAB outputSet Generated Tests Output\n');
fprintf(fid, '========================================\n\n');

%% Test 1: No output equation (y = x)
fprintf(fid, 'Test 1: No output equation (y = x)\n');
fprintf(fid, '-----------------------------------\n');
A = [-1, 0; 0, -2];
B = [1; 1];
% No C matrix means y = x
sys = linearSys(A, B);

% Create reachable set
R = zonotope([1; 1], 0.1*eye(2));

params = struct();
params.U = zonotope(0);
params.uTrans = zeros(1, 1);
params.V = zonotope(0);

options = struct();
options.compOutputSet = true;

Y = outputSet(sys, R, params, options);

fprintf(fid, 'Y.c = [%.15g; %.15g]\n', Y.c(1), Y.c(2));
fprintf(fid, 'Y.G shape = %dx%d\n', size(Y.G, 1), size(Y.G, 2));
fprintf(fid, 'Y.G = [');
for i = 1:size(Y.G, 1)
    for j = 1:size(Y.G, 2)
        fprintf(fid, '%.15g', Y.G(i, j));
        if j < size(Y.G, 2), fprintf(fid, ', '); end
    end
    if i < size(Y.G, 1), fprintf(fid, '; '); end
end
fprintf(fid, ']\n');
fprintf(fid, 'Y should equal R (no output equation)\n\n');

%% Test 2: With D matrix (direct feedthrough)
fprintf(fid, 'Test 2: With D matrix (direct feedthrough)\n');
fprintf(fid, '------------------------------------------\n');
A = [-1, 0; 0, -2];
B = [1; 1];
C = [1, 0];
D = 0.5;  % Direct feedthrough
sys = linearSys(A, B, [], C, D);

R = zonotope([1; 1], 0.1*eye(2));

params = struct();
params.U = zonotope(0.1, 0.05);
params.uTrans = 0.1;
params.V = zonotope(0);

options = struct();
options.compOutputSet = true;

Y = outputSet(sys, R, params, options);

fprintf(fid, 'Y.c = [%.15g]\n', Y.c(1));
fprintf(fid, 'Y.G shape = %dx%d\n', size(Y.G, 1), size(Y.G, 2));
fprintf(fid, 'Y.G = [');
for i = 1:size(Y.G, 1)
    for j = 1:size(Y.G, 2)
        fprintf(fid, '%.15g', Y.G(i, j));
        if j < size(Y.G, 2), fprintf(fid, ', '); end
    end
    if i < size(Y.G, 1), fprintf(fid, '; '); end
end
fprintf(fid, ']\n\n');

%% Test 3: compOutputSet false
fprintf(fid, 'Test 3: compOutputSet false\n');
fprintf(fid, '---------------------------\n');
A = [-1, 0; 0, -2];
B = [1; 1];
C = [1, 0];
sys = linearSys(A, B, [], C);

R = zonotope([1; 1], 0.1*eye(2));

params = struct();
params.U = zonotope(0);
params.uTrans = zeros(1, 1);
params.V = zonotope(0);

options = struct();
options.compOutputSet = false;  % Skip computation

Y = outputSet(sys, R, params, options);

fprintf(fid, 'Y should equal R (compOutputSet = false)\n');
fprintf(fid, 'Y.c = [%.15g; %.15g]\n', Y.c(1), Y.c(2));
fprintf(fid, 'Y.G shape = %dx%d\n', size(Y.G, 1), size(Y.G, 2));
fprintf(fid, 'Y.G = [');
for i = 1:size(Y.G, 1)
    for j = 1:size(Y.G, 2)
        fprintf(fid, '%.15g', Y.G(i, j));
        if j < size(Y.G, 2), fprintf(fid, ', '); end
    end
    if i < size(Y.G, 1), fprintf(fid, '; '); end
end
fprintf(fid, ']\n\n');

%% Test 4: With order reduction
fprintf(fid, 'Test 4: With order reduction\n');
fprintf(fid, '----------------------------\n');
A = [-1, 0; 0, -2];
B = [1; 1];
C = [1, 0];
sys = linearSys(A, B, [], C);

R = zonotope([1; 1], 0.1*eye(2));

params = struct();
params.U = zonotope(0);
params.uTrans = zeros(1, 1);
params.V = zonotope(0);

options = struct();
options.compOutputSet = true;
options.saveOrder = 5;
options.reductionTechnique = 'girard';

Y = outputSet(sys, R, params, options);

fprintf(fid, 'Y.c = [%.15g]\n', Y.c(1));
fprintf(fid, 'Y.G shape = %dx%d\n', size(Y.G, 1), size(Y.G, 2));
fprintf(fid, 'Y.G = [');
for i = 1:size(Y.G, 1)
    for j = 1:size(Y.G, 2)
        fprintf(fid, '%.15g', Y.G(i, j));
        if j < size(Y.G, 2), fprintf(fid, ', '); end
    end
    if i < size(Y.G, 1), fprintf(fid, '; '); end
end
fprintf(fid, ']\n');
fprintf(fid, 'Y should have reduced order (saveOrder = 5)\n\n');

fclose(fid);
fprintf('MATLAB outputSet generated tests completed. Results saved to matlab_outputSet_generated_output.txt\n');
