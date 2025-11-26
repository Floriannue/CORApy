% Debug script to understand pagemtimes behavior for evaluateSensitivity
% Compare with Python test expectations

% Test case matching Python test
S = cat(3, [1, 2; 3, 4], [5, 6; 7, 8]);  % Shape: (2, 2, 2) = (batch_size, input_dim, input_dim)
W = [1, 2; 3, 4; 5, 6];  % Shape: (3, 2) = (output_dim, input_dim)

fprintf('S shape: %s\n', mat2str(size(S)));
fprintf('W shape: %s\n', mat2str(size(W)));
fprintf('W.T shape: %s\n', mat2str(size(W.')));

% Test pagemtimes(S, W)
result1 = pagemtimes(S, W);
fprintf('\npagemtimes(S, W) shape: %s\n', mat2str(size(result1)));
fprintf('pagemtimes(S, W) result:\n');
disp(result1);

% Test manual computation: S[0] @ W.T
S0 = S(:, :, 1);
manual1 = S0 * W.';
fprintf('\nS(:, :, 1) shape: %s\n', mat2str(size(S0)));
fprintf('S(:, :, 1) * W.T shape: %s\n', mat2str(size(manual1)));
fprintf('S(:, :, 1) * W.T result:\n');
disp(manual1);

% Test pagemtimes(S, W.')
result2 = pagemtimes(S, W.');
fprintf('\npagemtimes(S, W.T) shape: %s\n', mat2str(size(result2)));
fprintf('pagemtimes(S, W.T) result:\n');
disp(result2);

% Check if result1 matches manual computation
fprintf('\n--- Comparison ---\n');
fprintf('Does pagemtimes(S, W) match S[:,:,1] * W.T?\n');
fprintf('result1(:, :, 1):\n');
disp(result1(:, :, 1));
fprintf('manual1:\n');
disp(manual1);
fprintf('Are they equal? %d\n', isequal(result1(:, :, 1), manual1));

% Test with actual sensitivity matrix shape from calcSensitivity
% S should be (nK, output_dim, bSz) where nK is number of generators
fprintf('\n--- Actual sensitivity shape from calcSensitivity ---\n');
nK = 2;  % number of generators
output_dim = 3;  % output dimension of this layer
bSz = 2;  % batch size
S_actual = rand(nK, output_dim, bSz);
fprintf('S_actual shape (nK, output_dim, bSz): %s\n', mat2str(size(S_actual)));
fprintf('W shape (output_dim, input_dim): %s\n', mat2str(size(W)));

result_actual = pagemtimes(S_actual, W);
fprintf('pagemtimes(S_actual, W) shape: %s\n', mat2str(size(result_actual)));
fprintf('Expected: (nK, input_dim, bSz) = (%d, %d, %d)\n', nK, size(W, 2), bSz);

% Manual check for first batch
S_actual_b0 = S_actual(:, :, 1);  % (nK, output_dim)
manual_actual = S_actual_b0 * W;  % (nK, output_dim) @ (output_dim, input_dim) = (nK, input_dim)
fprintf('\nS_actual(:, :, 1) shape: %s\n', mat2str(size(S_actual_b0)));
fprintf('S_actual(:, :, 1) * W shape: %s\n', mat2str(size(manual_actual)));
fprintf('result_actual(:, :, 1) shape: %s\n', mat2str(size(result_actual(:, :, 1))));
fprintf('Are they equal? %d\n', isequal(result_actual(:, :, 1), manual_actual));

