% Generate exact input-output pairs for zonotope evaluation
% This will be used to fix the Python implementation

fid = fopen('matlab_zonotope_io_pairs.txt', 'w');

fprintf(fid, '=== MATLAB Zonotope Input-Output Pairs ===\n\n');

% Create the same Conv2D layer
W = zeros(2, 2, 1, 2);
W(:, :, 1, 1) = [1, -1; -1, 2];
W(:, :, 1, 2) = [2, 3; -1, -2];
b = [1.0; -2.0];

layer = nnConv2DLayer(W, b);
nn = neuralNetwork({layer});
n = 4;
nn.setInputSize([n, n, 1]);

% Input point
x = reshape(eye(n), [], 1);

fprintf(fid, 'Input point x (all %d values):\n', length(x));
fprintf(fid, '%f ', x);
fprintf(fid, '\n\n');

% Input zonotope
G = 0.01 * eye(n * n);
X = zonotope(x, G);

fprintf(fid, 'Input zonotope X:\n');
fprintf(fid, '  Center (all %d values): ', length(X.c));
fprintf(fid, '%f ', X.c);
fprintf(fid, '\n');
fprintf(fid, '  Generator matrix shape: [%d %d]\n', size(X.G));
fprintf(fid, '  Generator matrix (first 5 rows, all cols):\n');
for i = 1:min(5, size(X.G, 1))
    fprintf(fid, '    ');
    fprintf(fid, '%f ', X.G(i, :));
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

% Evaluate through network
options = struct('nn', struct('train', struct('backprop', false), 'use_dlconv', false));

% Evaluate point
y_point = nn.evaluate(x, options);
fprintf(fid, 'Output point y_point (all %d values):\n', length(y_point));
fprintf(fid, '%f ', y_point);
fprintf(fid, '\n\n');

% Evaluate zonotope
Y = nn.evaluate(X, options);
fprintf(fid, 'Output zonotope Y:\n');
fprintf(fid, '  Center (all %d values): ', length(Y.c));
fprintf(fid, '%f ', Y.c);
fprintf(fid, '\n');
fprintf(fid, '  Generator matrix shape: [%d %d]\n', size(Y.G));
fprintf(fid, '  Generator matrix (first 5 rows, all cols):\n');
for i = 1:min(5, size(Y.G, 1))
    fprintf(fid, '    ');
    fprintf(fid, '%f ', Y.G(i, :));
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

% Verify center matches point
fprintf(fid, 'Verification:\n');
center_diff = Y.c - y_point;
fprintf(fid, '  Center - Point (all %d values): ', length(center_diff));
fprintf(fid, '%e ', center_diff);
fprintf(fid, '\n');
fprintf(fid, '  Max absolute difference: %e\n', max(abs(center_diff)));
fprintf(fid, '  Center matches point: %d\n', all(abs(center_diff) < 1e-10));
fprintf(fid, '\n');

% Compute expected center by converting to linear layer
% Convert the conv layer to linear layer to get Wff and bias
lin_layer = layer.convert2nnLinearLayer();
Wff = lin_layer.W;
bias = lin_layer.b;

fprintf(fid, 'Weight matrix Wff (from convert2nnLinearLayer):\n');
fprintf(fid, '  Shape: [%d %d]\n', size(Wff));
fprintf(fid, '  Wff (first 5 rows, all cols):\n');
for i = 1:min(5, size(Wff, 1))
    fprintf(fid, '    ');
    fprintf(fid, '%f ', Wff(i, :));
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

fprintf(fid, 'Bias (from convert2nnLinearLayer):\n');
fprintf(fid, '  Shape: [%d %d]\n', size(bias));
fprintf(fid, '  Bias (all %d values): ', length(bias));
fprintf(fid, '%f ', bias);
fprintf(fid, '\n\n');

expected_center = Wff * x + bias;
fprintf(fid, 'Expected center (Wff * x + bias):\n');
fprintf(fid, '  All %d values: ', length(expected_center));
fprintf(fid, '%f ', expected_center);
fprintf(fid, '\n');
fprintf(fid, '  Matches Y.c: %d\n', all(abs(expected_center - Y.c) < 1e-10));
fprintf(fid, '  Matches y_point: %d\n', all(abs(expected_center - y_point) < 1e-10));
fprintf(fid, '\n');

fclose(fid);
fprintf('Results saved to matlab_zonotope_io_pairs.txt\n');

