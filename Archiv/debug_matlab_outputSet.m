% Debug script to verify outputSet test against MATLAB
% This generates exact input/output pairs for Python tests

fprintf('=== Test 1: Basic Functionality ===\n\n');

% Setup: Create a linear system with output equation (matching Python test)
A = [-1, 0; 0, -2];
B = [1; 1];
C = [1, 0];  % Output: y = x1
sys = linearSys(A, B, [], C);

fprintf('Input:\n');
fprintf('A = [%g, %g; %g, %g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf('B = [%g; %g]\n', B(1), B(2));
fprintf('C = [%g, %g]\n', C(1), C(2));

% Create a reachable set (single set, not ReachSet)
R = zonotope([1; 1], 0.1 * eye(2));

fprintf('R center = [%g; %g]\n', R.c(1), R.c(2));
fprintf('R generators shape: %dx%d\n', size(R.G,1), size(R.G,2));

params.U = zonotope(zeros(1,1), []);
params.uTrans = zeros(1,1);
params.V = zonotope(zeros(1,1), []);

options.compOutputSet = true;

fprintf('params.U center = [%g]\n', params.U.c(1));
fprintf('params.uTrans = [%g]\n', params.uTrans(1));
fprintf('params.V center = [%g]\n', params.V.c(1));
fprintf('options.compOutputSet = %d\n', options.compOutputSet);

% Execute
Y = outputSet(sys, R, params, options);
Verror = 0;  % MATLAB outputSet doesn't return Verror, it's always 0 for linear systems

fprintf('\n=== Output Results ===\n');
fprintf('Y type: %s\n', class(Y));
if isa(Y, 'zonotope')
    fprintf('Y center = [%.15g]\n', Y.c(1));
    fprintf('Y generators shape: %dx%d\n', size(Y.G,1), size(Y.G,2));
    if size(Y.G,2) > 0
        fprintf('Y.G = [%.15g', Y.G(1,1));
        for i=2:size(Y.G,2)
            fprintf(', %.15g', Y.G(1,i));
        end
        fprintf(']\n');
    end
elseif isa(Y, 'interval')
    fprintf('Y.inf = [%.15g]\n', Y.inf(1));
    fprintf('Y.sup = [%.15g]\n', Y.sup(1));
end
fprintf('Verror = %d (always 0 for linear systems)\n', Verror);

% Save to file
fid = fopen('outputSet_matlab_output.txt', 'w');
fprintf(fid, 'MATLAB outputSet Test Output\n');
fprintf(fid, '============================\n\n');
fprintf(fid, 'Input:\n');
fprintf(fid, 'A = [%g, %g; %g, %g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf(fid, 'B = [%g; %g]\n', B(1), B(2));
fprintf(fid, 'C = [%g, %g]\n', C(1), C(2));
fprintf(fid, 'R center = [%g; %g]\n', R.c(1), R.c(2));
fprintf(fid, 'R generators shape: %dx%d\n', size(R.G,1), size(R.G,2));
fprintf(fid, '\nOutput:\n');
fprintf(fid, 'Y type: %s\n', class(Y));
if isa(Y, 'zonotope')
    fprintf(fid, 'Y center = [%.15g]\n', Y.c(1));
    fprintf(fid, 'Y generators shape: %dx%d\n', size(Y.G,1), size(Y.G,2));
    if size(Y.G,2) > 0
        fprintf(fid, 'Y.G = [%.15g', Y.G(1,1));
        for i=2:size(Y.G,2)
            fprintf(fid, ', %.15g', Y.G(1,i));
        end
        fprintf(fid, ']\n');
    end
elseif isa(Y, 'interval')
    fprintf(fid, 'Y.inf = [%.15g]\n', Y.inf(1));
    fprintf(fid, 'Y.sup = [%.15g]\n', Y.sup(1));
end
fprintf(fid, 'Verror = %d (always 0 for linear systems)\n', Verror);
fclose(fid);

fprintf('\nResults saved to outputSet_matlab_output.txt\n');
