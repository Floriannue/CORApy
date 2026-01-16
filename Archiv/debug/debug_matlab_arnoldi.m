% Debug script to verify arnoldi test against MATLAB
% This generates exact input/output pairs for Python tests

fprintf('=== Test 4: Small System (2x2) ===\n\n');

% Setup: Create a 2x2 system (matching Python test)
A = [1, 2; 3, 4];
vInit = [1; 0];
redDim = 2;

fprintf('Input:\n');
fprintf('A = [%g, %g; %g, %g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf('vInit = [%g; %g]\n', vInit(1), vInit(2));
fprintf('redDim = %d\n', redDim);

% Execute
[V, H, Hlast, happyBreakdown] = arnoldi(A, vInit, redDim);

fprintf('\n=== Output Results ===\n');
fprintf('V shape: %dx%d\n', size(V,1), size(V,2));
fprintf('V = \n');
disp(V);
fprintf('H shape: %dx%d\n', size(H,1), size(H,2));
fprintf('H = \n');
disp(H);
fprintf('Hlast = %.15g\n', Hlast);
fprintf('happyBreakdown = %d\n', happyBreakdown);

% Check orthonormality
VTV = V' * V;
fprintf('\nV''*V (should be identity):\n');
disp(VTV);
maxErr = max(max(abs(VTV - eye(size(V,2)))));
fprintf('Max error from identity: %.15g\n', maxErr);

% Save to file
fid = fopen('arnoldi_matlab_output.txt', 'w');
fprintf(fid, 'MATLAB arnoldi Test Output - Small System\n');
fprintf(fid, '==========================================\n\n');
fprintf(fid, 'Input:\n');
fprintf(fid, 'A = [%g, %g; %g, %g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf(fid, 'vInit = [%g; %g]\n', vInit(1), vInit(2));
fprintf(fid, 'redDim = %d\n', redDim);
fprintf(fid, '\nOutput:\n');
fprintf(fid, 'V shape: %dx%d\n', size(V,1), size(V,2));
fprintf(fid, 'V = [%.15g, %.15g; %.15g, %.15g]\n', V(1,1), V(1,2), V(2,1), V(2,2));
fprintf(fid, 'H shape: %dx%d\n', size(H,1), size(H,2));
fprintf(fid, 'H = [%.15g, %.15g; %.15g, %.15g]\n', H(1,1), H(1,2), H(2,1), H(2,2));
fprintf(fid, 'Hlast = %.15g\n', Hlast);
fprintf(fid, 'happyBreakdown = %d\n', happyBreakdown);
fprintf(fid, 'V''*V max error: %.15g\n', maxErr);
fclose(fid);

fprintf('\nResults saved to arnoldi_matlab_output.txt\n');
