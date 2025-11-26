% Debug script to compare MATLAB DLToolbox vs ONNX Runtime evaluation
% This matches the test structure

% Add CORA to path
addpath(genpath('cora_matlab'));

% Test convolutional network
model_path = 'models/Cora/nn/unitTests/vnn_verivital_avgpool.onnx';

% Load network as CORA network
nn = neuralNetwork.readONNXNetwork(model_path, false, 'BCSS');

% Load network as DLToolbox network (equivalent to ONNX Runtime)
nn_dlt = nn.convertToDLToolboxNetwork();

% Test input
x = ones(nn.neurons_in, 1);
fprintf('Input shape: [%d, %d]\n', size(x, 1), size(x, 2));
fprintf('Input (first 10): ');
fprintf('%f ', x(1:min(10, length(x))));
fprintf('\n\n');

% CORA evaluation
y_cora = nn.evaluate(x);
fprintf('CORA output shape: [%d, %d]\n', size(y_cora, 1), size(y_cora, 2));
fprintf('CORA output: ');
fprintf('%f ', y_cora);
fprintf('\n\n');

% DLToolbox evaluation (equivalent to ONNX Runtime)
% MATLAB: nn_dlt.predict(reshape(x, nn.layers{1}.inputSize))'
input_shape = nn.layers{1}.inputSize;
fprintf('Input shape from first layer: [');
fprintf('%d ', input_shape);
fprintf(']\n');

x_reshaped = reshape(x, input_shape);
fprintf('Reshaped input shape: [');
fprintf('%d ', size(x_reshaped));
fprintf(']\n');

y_dlt = nn_dlt.predict(x_reshaped)';
fprintf('DLToolbox output shape: [%d, %d]\n', size(y_dlt, 1), size(y_dlt, 2));
fprintf('DLToolbox output: ');
fprintf('%f ', y_dlt);
fprintf('\n\n');

% Compare
tol = 1e-6;
diff = abs(y_cora - y_dlt);
fprintf('Difference: ');
fprintf('%e ', diff);
fprintf('\n');
fprintf('Max difference: %e\n', max(diff));
fprintf('Within tolerance: %d\n', all(withinTol(y_cora, y_dlt, tol)));

