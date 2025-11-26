% Debug script to understand how MATLAB verify computes sensitivity
% Compare with Python implementation

% Load the network and setup (matching Python example)
modelPath = 'cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx';
specPath = 'cora_python/examples/nn/models/prop_1.vnnlib';

% Load model
nn = neuralNetwork.readONNXNetwork(modelPath, false, 'BSSC');

% Load specification
[X0, specs] = vnnlib2cora(specPath);

% Compute center and radius
x = 1/2 * (X0{1}.sup + X0{1}.inf);
r = 1/2 * (X0{1}.sup - X0{1}.inf);

% Extract specification
if isa(specs.set, 'halfspace')
    A = specs.set.c.';
    b = -specs.set.d;
else
    A = specs.set.A;
    b = -specs.set.b;
end

safeSet = (specs.type == 'safeSet');

% Create options
options = struct();
options.nn = struct();
options.nn.use_approx_error = true;
options.nn.poly_method = 'bounds';
options.nn.num_generators = 100;
options.nn.train = struct();
options.nn.train.backprop = false;
options.nn.train.mini_batch_size = 512;
options.nn.interval_center = false;
options = nnHelper.validateNNoptions(options, true);

% Test sensitivity computation for single batch
fprintf('=== Single batch sensitivity ===\n');
xi_single = x;  % Shape: (5, 1)
fprintf('xi_single shape: %s\n', mat2str(size(xi_single)));

[S_single, y_single] = nn.calcSensitivity(xi_single, options, false);
fprintf('S_single shape: %s\n', mat2str(size(S_single)));
fprintf('y_single shape: %s\n', mat2str(size(y_single)));

% Compute sens from S_single
S_single = max(S_single, 1e-3);
sens_sum_single = sum(abs(S_single), 1);  % Sum over output dimension
fprintf('sens_sum_single shape (after sum): %s\n', mat2str(size(sens_sum_single)));
sens_single = permute(sens_sum_single, [2, 1, 3]);  % permute([2 1 3])
fprintf('sens_single shape (after permute): %s\n', mat2str(size(sens_single)));
sens_single = sens_single(:,:);  % Flatten to 2D
fprintf('sens_single shape (after flatten): %s\n', mat2str(size(sens_single)));

% Test sensitivity computation for multiple batches
fprintf('\n=== Multiple batch sensitivity ===\n');
% Create multiple batches by repeating x
xi_multi = repmat(x, 1, 5);  % Shape: (5, 5)
fprintf('xi_multi shape: %s\n', mat2str(size(xi_multi)));

[S_multi, y_multi] = nn.calcSensitivity(xi_multi, options, false);
fprintf('S_multi shape: %s\n', mat2str(size(S_multi)));
fprintf('y_multi shape: %s\n', mat2str(size(y_multi)));

% Compute sens from S_multi
S_multi = max(S_multi, 1e-3);
sens_sum_multi = sum(abs(S_multi), 1);  % Sum over output dimension
fprintf('sens_sum_multi shape (after sum): %s\n', mat2str(size(sens_sum_multi)));
sens_multi = permute(sens_sum_multi, [2, 1, 3]);  % permute([2 1 3])
fprintf('sens_multi shape (after permute): %s\n', mat2str(size(sens_multi)));
sens_multi = sens_multi(:,:);  % Flatten to 2D
fprintf('sens_multi shape (after flatten): %s\n', mat2str(size(sens_multi)));

% Test with larger batch size
fprintf('\n=== Large batch sensitivity ===\n');
xi_large = repmat(x, 1, 512);  % Shape: (5, 512)
fprintf('xi_large shape: %s\n', mat2str(size(xi_large)));

[S_large, y_large] = nn.calcSensitivity(xi_large, options, false);
fprintf('S_large shape: %s\n', mat2str(size(S_large)));
fprintf('y_large shape: %s\n', mat2str(size(y_large)));

% Compute sens from S_large
S_large = max(S_large, 1e-3);
sens_sum_large = sum(abs(S_large), 1);  % Sum over output dimension
fprintf('sens_sum_large shape (after sum): %s\n', mat2str(size(sens_sum_large)));
sens_large = permute(sens_sum_large, [2, 1, 3]);  % permute([2 1 3])
fprintf('sens_large shape (after permute): %s\n', mat2str(size(sens_large)));
sens_large = sens_large(:,:);  % Flatten to 2D
fprintf('sens_large shape (after flatten): %s\n', mat2str(size(sens_large)));

fprintf('\n=== Summary ===\n');
fprintf('For single batch: sens should be (1, 5)\n');
fprintf('For 5 batches: sens should be (5, 5)\n');
fprintf('For 512 batches: sens should be (512, 5)\n');

