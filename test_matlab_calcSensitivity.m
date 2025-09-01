% Test script for calcSensitivity to understand the dimensions
% This matches the Python test case

% Set random seed for reproducibility
rng(42);

% Network dimensions (matching Python test)
n0 = 5;  % input dimension
nK = 7;  % output dimension
bSz = 13; % batch size

% Generate a random neural network
nn = neuralNetwork.generateRandom( ...
    'NrInputs', n0, ...
    'NrOutputs', nK, ...
    'NrLayers', 3, ...
    'NrHiddenNeurons', 17 ...
);

% Append a softmax layer
nn.layers{end+1} = nnSoftmaxLayer;

fprintf('Network has %d layers\n', length(nn.layers));
for i = 1:length(nn.layers)
    fprintf('Layer %d: %s\n', i, class(nn.layers{i}));
end

% Generate random input
x = rand([n0 bSz]);

% Compute output
y = nn.evaluate(x);
fprintf('Output shape: %s\n', mat2str(size(y)));

% Calculate sensitivity
S = nn.calcSensitivity(x, struct, true);
fprintf('Final sensitivity S shape: %s\n', mat2str(size(S)));

% Check sensitivity at each layer
fprintf('\nSensitivity at each layer:\n');
for i = 1:length(nn.layers)
    if isfield(nn.layers{i}, 'sensitivity')
        fprintf('Layer %d (%s): %s\n', i, class(nn.layers{i}), mat2str(size(nn.layers{i}.sensitivity)));
    else
        fprintf('Layer %d (%s): no sensitivity stored\n', i, class(nn.layers{i}));
    end
end

% Test the dx computation
x_ = rand([n0 bSz]);
dx = x - x_;
fprintf('\ndx shape: %s\n', mat2str(size(dx)));

% MATLAB way: dy = pagemtimes(S,permute(dx,[1 3 2]));
dx_permuted = permute(dx, [1 3 2]);
fprintf('dx_permuted shape: %s\n', mat2str(size(dx_permuted)));

dy = pagemtimes(S, dx_permuted);
fprintf('dy shape: %s\n', mat2str(size(dy)));

% Compute new output
y_ = nn.evaluate(x + dx);
fprintf('y_ shape: %s\n', mat2str(size(y_)));

% Check if directions are correct
sign_check = all(sign(y + dy) == sign(y_), 'all');
fprintf('Sign check passed: %s\n', mat2str(sign_check));
