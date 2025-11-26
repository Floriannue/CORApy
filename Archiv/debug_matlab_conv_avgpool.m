% Debug script to get expected values for Conv2D and AvgPool2D layers
% Output to file for Python comparison

% Add CORA to path
addpath(genpath('cora_matlab'));

% Open output file
fid = fopen('matlab_conv_avgpool_output.txt', 'w');

fprintf(fid, '=== Testing Conv2D Layer ===\n\n');

% Load network
model_path = 'models/Cora/nn/unitTests/vnn_verivital_avgpool.onnx';
nn = neuralNetwork.readONNXNetwork(model_path, false, 'BCSS');

% Get Conv2D layer (first layer)
conv_layer = nn.layers{1};
fprintf(fid, 'Conv2D Layer:\n');
fprintf(fid, '  inputSize: [%d %d %d]\n', conv_layer.inputSize(1), conv_layer.inputSize(2), conv_layer.inputSize(3));
fprintf(fid, '  W shape: [%d %d %d %d]\n', size(conv_layer.W, 1), size(conv_layer.W, 2), size(conv_layer.W, 3), size(conv_layer.W, 4));
fprintf(fid, '  b shape: [%d]\n', length(conv_layer.b));
fprintf(fid, '  stride: [%d %d]\n', conv_layer.stride(1), conv_layer.stride(2));
fprintf(fid, '  padding: [%d %d %d %d]\n', conv_layer.padding(1), conv_layer.padding(2), conv_layer.padding(3), conv_layer.padding(4));
fprintf(fid, '  dilation: [%d %d]\n', conv_layer.dilation(1), conv_layer.dilation(2));
fprintf(fid, '\n');

% Print W and b values (first few)
fprintf(fid, '  W (first filter, first channel):\n');
fprintf(fid, '    [%f %f]\n', conv_layer.W(1,1,1,1), conv_layer.W(1,2,1,1));
fprintf(fid, '    [%f %f]\n', conv_layer.W(2,1,1,1), conv_layer.W(2,2,1,1));
fprintf(fid, '  b (first 10): ');
fprintf(fid, '%f ', conv_layer.b(1:min(10, length(conv_layer.b))));
fprintf(fid, '\n\n');

% Create test input (all ones, reshaped to inputSize)
x = ones(conv_layer.inputSize(1) * conv_layer.inputSize(2) * conv_layer.inputSize(3), 1);
fprintf(fid, 'Test input:\n');
fprintf(fid, '  x shape: [%d %d]\n', size(x, 1), size(x, 2));
fprintf(fid, '  x (first 10): ');
fprintf(fid, '%f ', x(1:min(10, length(x))));
fprintf(fid, '\n\n');

% Evaluate through network layer by layer using evaluate_ with idxLayer
% This allows us to get intermediate outputs
% Use evaluate method which properly initializes options via validateNNoptions
% For intermediate outputs, we'll use evaluate with idxLayer parameter
% But evaluate doesn't support idxLayer directly, so we need to use evaluate_
% with properly validated options
options = nnHelper.validateNNoptions(struct());

% Evaluate only first layer (Conv2D)
y_conv = nn.evaluate_(x, options, 1);
fprintf(fid, 'Conv2D output:\n');
fprintf(fid, '  y_conv shape: [%d %d]\n', size(y_conv, 1), size(y_conv, 2));
fprintf(fid, '  y_conv (first 50): ');
fprintf(fid, '%f ', y_conv(1:min(50, length(y_conv))));
fprintf(fid, '\n');
fprintf(fid, '  y_conv (last 10): ');
fprintf(fid, '%f ', y_conv(max(1, length(y_conv)-9):end));
fprintf(fid, '\n\n');

% Evaluate first two layers (Conv2D + ReLU)
y_relu = nn.evaluate_(x, options, [1 2]);
fprintf(fid, 'ReLU output (first 50): ');
fprintf(fid, '%f ', y_relu(1:min(50, length(y_relu))));
fprintf(fid, '\n\n');

% Get AvgPool2D layer (third layer)
avgpool_layer = nn.layers{3};
fprintf(fid, '=== Testing AvgPool2D Layer ===\n\n');
fprintf(fid, 'AvgPool2D Layer:\n');
fprintf(fid, '  inputSize: [%d %d %d]\n', avgpool_layer.inputSize(1), avgpool_layer.inputSize(2), avgpool_layer.inputSize(3));
fprintf(fid, '  W shape: [%d %d %d %d]\n', size(avgpool_layer.W, 1), size(avgpool_layer.W, 2), size(avgpool_layer.W, 3), size(avgpool_layer.W, 4));
fprintf(fid, '  stride: [%d %d]\n', avgpool_layer.stride(1), avgpool_layer.stride(2));
fprintf(fid, '  padding: [%d %d %d %d]\n', avgpool_layer.padding(1), avgpool_layer.padding(2), avgpool_layer.padding(3), avgpool_layer.padding(4));
fprintf(fid, '\n');

% Evaluate first three layers (Conv2D + ReLU + AvgPool2D)
y_avgpool = nn.evaluate_(x, options, [1 2 3]);
fprintf(fid, 'AvgPool2D output:\n');
fprintf(fid, '  y_avgpool shape: [%d %d]\n', size(y_avgpool, 1), size(y_avgpool, 2));
fprintf(fid, '  y_avgpool (first 50): ');
fprintf(fid, '%f ', y_avgpool(1:min(50, length(y_avgpool))));
fprintf(fid, '\n');
fprintf(fid, '  y_avgpool (last 10): ');
fprintf(fid, '%f ', y_avgpool(max(1, length(y_avgpool)-9):end));
fprintf(fid, '\n\n');

% Test with different input values
fprintf(fid, '=== Testing with random input ===\n\n');
rng('default');
x_rand = rand(conv_layer.inputSize(1) * conv_layer.inputSize(2) * conv_layer.inputSize(3), 1);
fprintf(fid, 'Random input (first 10): ');
fprintf(fid, '%f ', x_rand(1:min(10, length(x_rand))));
fprintf(fid, '\n');
fprintf(fid, 'Random input (all %d values): ', length(x_rand));
fprintf(fid, '%f ', x_rand);
fprintf(fid, '\n\n');

y_conv_rand = nn.evaluate_(x_rand, options, 1);
fprintf(fid, 'Conv2D output with random input (first 50): ');
fprintf(fid, '%f ', y_conv_rand(1:min(50, length(y_conv_rand))));
fprintf(fid, '\n\n');

y_relu_rand = nn.evaluate_(x_rand, options, [1 2]);
y_avgpool_rand = nn.evaluate_(x_rand, options, [1 2 3]);
fprintf(fid, 'AvgPool2D output with random input (first 50): ');
fprintf(fid, '%f ', y_avgpool_rand(1:min(50, length(y_avgpool_rand))));
fprintf(fid, '\n\n');

fclose(fid);
fprintf('Results saved to matlab_conv_avgpool_output.txt\n');
