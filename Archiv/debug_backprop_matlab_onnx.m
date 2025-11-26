% Debug script to compare MATLAB DLToolbox vs ONNX Runtime
% This helps us understand what the correct output should be

% Load ONNX model with DLToolbox
model_path = 'cora_matlab/models/Cora/nn/nn-nav-set.onnx';
nn_dlt = importNetworkFromONNX(model_path);

% Load CORA network
nn = neuralNetwork.readONNXNetwork(model_path);

% Test input
x = ones(nn.neurons_in, 1);

% CORA evaluation
y_cora = nn.evaluate(x);
fprintf('CORA output: [%.8f, %.8f]\n', y_cora(1), y_cora(2));

% DLToolbox evaluation (matches MATLAB test)
y_dlt = nn_dlt.predict(x')';
fprintf('DLToolbox output: [%.8f, %.8f]\n', y_dlt(1), y_dlt(2));

% Check if they match
tol = 1e-6;
match = all(withinTol(y_cora, y_dlt, tol));
fprintf('Match (tol=1e-6): %d\n', match);
if ~match
    fprintf('Difference: [%.8f, %.8f]\n', y_cora(1)-y_dlt(1), y_cora(2)-y_dlt(2));
end

% Check network structure
fprintf('\nDLToolbox network structure:\n');
fprintf('  Input size: %s\n', mat2str(nn_dlt.Layers(1).InputSize));
fprintf('  Number of layers: %d\n', length(nn_dlt.Layers));
for i=1:min(5, length(nn_dlt.Layers))
    fprintf('  Layer %d: %s\n', i, class(nn_dlt.Layers(i)));
end

fprintf('\nCORA network structure:\n');
fprintf('  neurons_in: %d\n', nn.neurons_in);
fprintf('  neurons_out: %d\n', nn.neurons_out);
fprintf('  Number of layers: %d\n', length(nn.layers));
for i=1:min(5, length(nn.layers))
    fprintf('  Layer %d: %s\n', i, class(nn.layers{i}));
end

