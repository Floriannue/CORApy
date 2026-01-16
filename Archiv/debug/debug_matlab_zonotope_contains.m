% Debug script to test zonotope containment after Conv2D layer evaluation
% Compare with Python implementation

% Open output file
fid = fopen('matlab_zonotope_contains_output.txt', 'w');

fprintf(fid, '=== Testing Zonotope Containment after Conv2D Layer ===\n\n');

% Create the same Conv2D layer as in Python test
W = zeros(2, 2, 1, 2);
W(:, :, 1, 1) = [1, -1; -1, 2];  % filter 1
W(:, :, 1, 2) = [2, 3; -1, -2];  % filter 2
b = [1.0; -2.0];

layer = nnConv2DLayer(W, b);

nn = neuralNetwork({layer});
n = 4;
nn.setInputSize([n, n, 1]);

fprintf(fid, 'Layer configuration:\n');
fprintf(fid, '  Input size: [%d %d %d]\n', n, n, 1);
fprintf(fid, '  W shape: [%d %d %d %d]\n', size(W));
fprintf(fid, '  b shape: [%d %d]\n', size(b));
fprintf(fid, '\n');

% MATLAB: x = reshape(eye(n),[],1);
x = reshape(eye(n), [], 1);

fprintf(fid, 'Input point x:\n');
fprintf(fid, '  Shape: [%d %d]\n', size(x));
fprintf(fid, '  First 10 values: ');
fprintf(fid, '%f ', x(1:min(10, length(x))));
fprintf(fid, '\n\n');

% MATLAB: X = zonotope(x,0.01 * eye(n*n));
G = 0.01 * eye(n * n);
X = zonotope(x, G);

fprintf(fid, 'Input zonotope X:\n');
fprintf(fid, '  Center shape: [%d %d]\n', size(X.c));
fprintf(fid, '  Generator shape: [%d %d]\n', size(X.G));
fprintf(fid, '  Number of generators: %d\n', size(X.G, 2));
fprintf(fid, '\n');

% Evaluate zonotope through network
fprintf(fid, 'Evaluating zonotope through network...\n');
options = struct('nn', struct('train', struct('backprop', false), 'use_dlconv', false));
Y = nn.evaluate(X, options);

fprintf(fid, 'Output zonotope Y:\n');
fprintf(fid, '  Center shape: [%d %d]\n', size(Y.c));
fprintf(fid, '  Generator shape: [%d %d]\n', size(Y.G));
fprintf(fid, '  Number of generators: %d\n', size(Y.G, 2));
fprintf(fid, '  Center (first 10): ');
fprintf(fid, '%f ', Y.c(1:min(10, length(Y.c))));
fprintf(fid, '\n\n');

% Evaluate point through network
fprintf(fid, 'Evaluating point through network...\n');
y_point = nn.evaluate(x, options);

fprintf(fid, 'Output point y_point:\n');
fprintf(fid, '  Shape: [%d %d]\n', size(y_point));
fprintf(fid, '  First 10 values: ');
fprintf(fid, '%f ', y_point(1:min(10, length(y_point))));
fprintf(fid, '\n\n');

% Check if center matches
fprintf(fid, 'Center comparison:\n');
center_diff = Y.c - y_point;
fprintf(fid, '  Max difference: %e\n', max(abs(center_diff)));
fprintf(fid, '  All close (tol=1e-10): %d\n', all(abs(center_diff) < 1e-10));
fprintf(fid, '\n');

% Check containment
fprintf(fid, 'Testing containment: Y.contains(y_point)\n');
try
    contains_result = Y.contains(y_point);
    fprintf(fid, '  Result: %d\n', contains_result);
    fprintf(fid, '  Success: Containment check completed\n');
catch ME
    fprintf(fid, '  Error: %s\n', ME.message);
    fprintf(fid, '  Identifier: %s\n', ME.identifier);
    fprintf(fid, '  Stack:\n');
    for i = 1:length(ME.stack)
        fprintf(fid, '    %s at line %d\n', ME.stack(i).name, ME.stack(i).line);
    end
end
fprintf(fid, '\n');

% Additional diagnostics
fprintf(fid, 'Additional diagnostics:\n');
fprintf(fid, '  Y dimension: %d\n', length(Y.c));
fprintf(fid, '  Y number of generators: %d\n', size(Y.G, 2));
fprintf(fid, '  Is degenerate (dim > generators): %d\n', length(Y.c) > size(Y.G, 2));
fprintf(fid, '\n');

% Test with a simpler case: check if zero point is contained
fprintf(fid, 'Testing containment of zero point:\n');
zero_point = zeros(size(Y.c));
try
    contains_zero = Y.contains(zero_point);
    fprintf(fid, '  Y.contains(zero): %d\n', contains_zero);
catch ME
    fprintf(fid, '  Error: %s\n', ME.message);
end
fprintf(fid, '\n');

fclose(fid);
fprintf('Results saved to matlab_zonotope_contains_output.txt\n');

