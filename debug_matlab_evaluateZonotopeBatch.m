% MATLAB script to test evaluateZonotopeBatch and compare with Python
% This replicates the Python test: test_evaluateZonotopeBatch_default_all_layers

% Set random seed for reproducibility
rng(0);

% Create layers - matches Python test
W1 = [1.0, 2.0; 3.0, 4.0];
b1 = [0.5; 1.0];
layer1 = nnLinearLayer(W1, b1);

W2 = [2.0, -1.0];
b2 = [0.2];
layer2 = nnLinearLayer(W2, b2);

nn = neuralNetwork({layer1, layer2});

% Create input - matches Python test
% Python: c = np.array([[[1.0]], [[-1.0]]])
% Shape: (2, 1, 1) - (n, 1, batch)
c = zeros(2, 1, 1);
c(1, 1, 1) = 1.0;
c(2, 1, 1) = -1.0;

% Python: G = np.array([[[0.1, 0.0]], [[0.0, 0.2]]])
% Shape: (2, 2, 1) - (n, q, batch)
G = zeros(2, 2, 1);
G(1, 1, 1) = 0.1;
G(1, 2, 1) = 0.0;
G(2, 1, 1) = 0.0;
G(2, 2, 1) = 0.2;

% Evaluate
[result_c, result_G] = nn.evaluateZonotopeBatch(c, G);

% Display results
fprintf('MATLAB Results:\n');
fprintf('result_c shape: [%d, %d, %d]\n', size(result_c));
fprintf('result_c:\n');
disp(result_c);
fprintf('result_G shape: [%d, %d, %d]\n', size(result_G));
fprintf('result_G:\n');
disp(result_G);

% Save to file for comparison
save('matlab_evaluateZonotopeBatch_results.mat', 'result_c', 'result_G', 'c', 'G', 'W1', 'b1', 'W2', 'b2');

fprintf('\nResults saved to matlab_evaluateZonotopeBatch_results.mat\n');

