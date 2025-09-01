% Test script for softmax sensitivity calculation
% This matches the Python test case

% Set random seed for reproducibility
rng(42);

% Network dimensions (matching Python test)
n0 = 5;  % input dimension
nK = 7;  % output dimension
bSz = 13; % batch size

% Generate random weights (matching Python test)
W1 = 2*rand(10, n0) - 1;
b1 = rand(10, 1);
W2 = 2*rand(nK, 10) - 1;
b2 = rand(nK, 1);

% Generate random input
x = rand(n0, bSz);

% Forward pass through first two layers
y1 = W1 * x + b1;  % Shape: (10, 13)
y2 = W2 * y1 + b2; % Shape: (7, 13)

% Apply softmax
y_softmax = exp(y2 - max(y2)) ./ sum(exp(y2 - max(y2))); % Shape: (7, 13)

% Initialize sensitivity matrix (like in calcSensitivity)
S = repmat(eye(nK), 1, 1, bSz); % Shape: (7, 7, 13)

fprintf('Initial S shape: %s\n', mat2str(size(S)));

% Now simulate what happens in softmax evaluateSensitivity
sx = permute(y_softmax, [1 3 2]); % Shape: (7, 1, 13)
fprintf('sx shape after permute: %s\n', mat2str(size(sx)));

% Compute Jacobian
J = pagemtimes(-sx, 'none', sx, 'transpose') + sx .* eye(size(sx,1));
fprintf('J shape: %s\n', mat2str(size(J)));

% Apply Jacobian to sensitivity matrix
S_new = pagemtimes(S, J);
fprintf('S_new shape after pagemtimes: %s\n', mat2str(size(S_new)));

% Check if dimensions are correct
fprintf('Expected S_new shape: (7, 7, 13)\n');
fprintf('Actual S_new shape: %s\n', mat2str(size(S_new)));

% Test with a simple case
fprintf('\n--- Simple test case ---\n');
S_simple = repmat(eye(2), 1, 1, 3); % Shape: (2, 2, 3)
J_simple = rand(2, 2, 3); % Shape: (2, 2, 3)
fprintf('S_simple shape: %s\n', mat2str(size(S_simple)));
fprintf('J_simple shape: %s\n', mat2str(size(J_simple)));

S_result = pagemtimes(S_simple, J_simple);
fprintf('S_result shape: %s\n', mat2str(size(S_result)));
