% Debug script to verify permute operation for zonotack beta_ computation
% This will help identify if the permute translation is correct

fprintf('=== Debug permute([2 4 1 3]) on 3D array ===\n\n');

% Create a test 3D array matching ld_Gyi(:,1:numInitGens,:) shape
% Shape: (p, numInitGens, cbSz) = (1, 2, 1) for our test case
p = 1;
numInitGens = 2;
cbSz = 1;

% Create test array
test_array = randn(p, numInitGens, cbSz);
fprintf('Input array shape: [%d %d %d]\n', size(test_array,1), size(test_array,2), size(test_array,3));
fprintf('Input array:\n');
disp(test_array);

% MATLAB permute([2 4 1 3])
% This means: new dim 1 = old dim 2, new dim 2 = old dim 4 (singleton), 
%             new dim 3 = old dim 1, new dim 4 = old dim 3
beta_ = -permute(sign(test_array), [2 4 1 3]);
fprintf('\nAfter -permute(sign(...), [2 4 1 3]):\n');
fprintf('Output shape: [%d %d %d %d]\n', size(beta_,1), size(beta_,2), size(beta_,3), size(beta_,4));
fprintf('Output array:\n');
disp(beta_);

% Reshape to (numInitGens, 1, p*cbSz)
beta = reshape(beta_, [numInitGens 1 p*cbSz]);
fprintf('\nAfter reshape to [numInitGens 1 p*cbSz] = [%d 1 %d]:\n', numInitGens, p*cbSz);
fprintf('Output shape: [%d %d %d]\n', size(beta,1), size(beta,2), size(beta,3));
fprintf('Output array:\n');
disp(beta);

fprintf('\n=== Expected Python equivalent ===\n');
fprintf('Python should produce the same result.\n');
fprintf('Check if the permute translation is correct.\n');

