% Debug script to verify arnoldi test against MATLAB
% This generates exact input/output pairs for Python tests
%
% Based on test_Krylov_Arnoldi.m and arnoldi.m

% Add CORA to path
addpath(genpath('cora_matlab'));

fid = fopen('matlab_arnoldi_output.txt', 'w');
fprintf(fid, 'MATLAB arnoldi Test Output\n');
fprintf(fid, '==========================\n\n');

%% Test 1: Basic functionality test
fprintf(fid, 'Test 1: Basic functionality test\n');
fprintf(fid, '---------------------------------\n');

% Setup: Create a simple test matrix (5x5)
% Use seed 42 for reproducibility (matching Python test)
rng(42);
A = randn(5, 5);
A = (A + A.') / 2;  % Make symmetric

% Print full A matrix for Python test
fprintf(fid, 'A full matrix:\n');
for i = 1:size(A, 1)
    fprintf(fid, 'A(%d,:) = [', i);
    for j = 1:size(A, 2)
        fprintf(fid, '%.15g', A(i,j));
        if j < size(A, 2)
            fprintf(fid, ', ');
        end
    end
    fprintf(fid, ']\n');
end

% Initial vector
vInit = randn(5, 1);
vInit = vInit / norm(vInit);  % Normalize

% Print full vInit for Python test
fprintf(fid, 'vInit full vector:\n');
fprintf(fid, 'vInit = [');
for i = 1:length(vInit)
    fprintf(fid, '%.15g', vInit(i));
    if i < length(vInit)
        fprintf(fid, '; ');
    end
end
fprintf(fid, ']\n');

% Reduced dimension
redDim = 3;

fprintf(fid, 'Input:\n');
fprintf(fid, 'A shape: %dx%d\n', size(A, 1), size(A, 2));
% Print full A matrix for Python test (exact values)
fprintf(fid, 'A full matrix (for Python test):\n');
fprintf(fid, 'A = np.array([\n');
for i = 1:size(A, 1)
    fprintf(fid, '    [');
    for j = 1:size(A, 2)
        fprintf(fid, '%.15g', A(i,j));
        if j < size(A, 2)
            fprintf(fid, ', ');
        end
    end
    fprintf(fid, ']');
    if i < size(A, 1)
        fprintf(fid, ',\n');
    else
        fprintf(fid, '\n');
    end
end
fprintf(fid, '])\n');
fprintf(fid, 'vInit shape: %dx%d\n', size(vInit, 1), size(vInit, 2));
% Print full vInit for Python test (exact values)
fprintf(fid, 'vInit full vector (for Python test):\n');
fprintf(fid, 'vInit = np.array([[');
for i = 1:length(vInit)
    fprintf(fid, '%.15g', vInit(i));
    if i < length(vInit)
        fprintf(fid, '], [');
    end
end
fprintf(fid, ']])\n');
fprintf(fid, 'redDim = %d\n', redDim);

% Execute
[V, H, Hlast, happyBreakdown] = arnoldi(A, vInit, redDim);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'V shape: %dx%d\n', size(V, 1), size(V, 2));
fprintf(fid, 'V(1,1) = %.15g\n', V(1,1));
fprintf(fid, 'V(2,1) = %.15g\n', V(2,1));
fprintf(fid, 'V(1,2) = %.15g\n', V(1,2));
fprintf(fid, 'H shape: %dx%d\n', size(H, 1), size(H, 2));
fprintf(fid, 'H(1,1) = %.15g\n', H(1,1));
fprintf(fid, 'H(1,2) = %.15g\n', H(1,2));
fprintf(fid, 'H(2,1) = %.15g\n', H(2,1));
fprintf(fid, 'Hlast = %.15g\n', Hlast);
fprintf(fid, 'happyBreakdown = %d\n', happyBreakdown);

% Verify orthonormality
VTV = V.' * V;
fprintf(fid, 'V.T @ V (should be identity):\n');
fprintf(fid, 'VTV(1,1) = %.15g\n', VTV(1,1));
fprintf(fid, 'VTV(1,2) = %.15g\n', VTV(1,2));
fprintf(fid, 'VTV(2,2) = %.15g\n', VTV(2,2));
fprintf(fid, '\n');

%% Test 2: Happy breakdown test
fprintf(fid, 'Test 2: Happy breakdown test\n');
fprintf(fid, '----------------------------\n');

% Setup: Create a matrix that will cause early termination
% Use a 3x3 identity-like matrix
A2 = eye(3);
vInit2 = [1; 0; 0];  % Already in the span
redDim2 = 5;  % Larger than dimension

fprintf(fid, 'Input:\n');
fprintf(fid, 'A2 = eye(3)\n');
fprintf(fid, 'vInit2 = [1; 0; 0]\n');
fprintf(fid, 'redDim2 = %d\n', redDim2);

% Execute
[V2, H2, Hlast2, happyBreakdown2] = arnoldi(A2, vInit2, redDim2);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'V2 shape: %dx%d\n', size(V2, 1), size(V2, 2));
fprintf(fid, 'H2 shape: %dx%d\n', size(H2, 1), size(H2, 2));
fprintf(fid, 'happyBreakdown2 = %d\n', happyBreakdown2);
fprintf(fid, '\n');

%% Test 3: Error handling test (zero vector)
fprintf(fid, 'Test 3: Error handling test (zero vector)\n');
fprintf(fid, '----------------------------------------\n');

A3 = randn(3, 3);
vInit3 = zeros(3, 1);
redDim3 = 3;

fprintf(fid, 'Input:\n');
fprintf(fid, 'vInit3 = zeros(3,1)\n');
fprintf(fid, 'redDim3 = %d\n', redDim3);

% Execute (should error)
try
    [V3, H3, Hlast3, happyBreakdown3] = arnoldi(A3, vInit3, redDim3);
    fprintf(fid, '\nOutput:\n');
    fprintf(fid, 'ERROR: Should have raised an error for zero vector\n');
catch ME
    fprintf(fid, '\nOutput:\n');
    fprintf(fid, 'Error raised: %s\n', ME.message);
    fprintf(fid, 'Error identifier: %s\n', ME.identifier);
end
fprintf(fid, '\n');

%% Test 4: Sparse matrix test
fprintf(fid, 'Test 4: Sparse matrix test\n');
fprintf(fid, '-------------------------\n');

% Create sparse matrix
A4 = sparse([1, 2, 3; 4, 5, 6; 7, 8, 9]);
vInit4 = [1; 1; 1] / sqrt(3);  % Normalized
redDim4 = 2;

fprintf(fid, 'Input:\n');
fprintf(fid, 'A4 is sparse: %d\n', issparse(A4));
fprintf(fid, 'A4 shape: %dx%d\n', size(A4, 1), size(A4, 2));
fprintf(fid, 'vInit4 shape: %dx%d\n', size(vInit4, 1), size(vInit4, 2));
fprintf(fid, 'redDim4 = %d\n', redDim4);

% Execute
[V4, H4, Hlast4, happyBreakdown4] = arnoldi(A4, vInit4, redDim4);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'V4 shape: %dx%d\n', size(V4, 1), size(V4, 2));
fprintf(fid, 'V4(1,1) = %.15g\n', V4(1,1));
fprintf(fid, 'H4 shape: %dx%d\n', size(H4, 1), size(H4, 2));
fprintf(fid, 'H4(1,1) = %.15g\n', H4(1,1));
fprintf(fid, 'happyBreakdown4 = %d\n', happyBreakdown4);
fprintf(fid, '\n');

fclose(fid);
fprintf('MATLAB arnoldi tests completed. Results saved to matlab_arnoldi_output.txt\n');
