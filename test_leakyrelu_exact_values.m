% Test LeakyReLU computeApproxError exact values
clear; clc;

% Create LeakyReLU layer
layer = nnLeakyReLULayer(0.01);

% Test case: m >= 1
l = -1;
u = 1;
coeffs = [1.0, 0];  % m = 1.0

fprintf('Testing LeakyReLU m >= 1 case:\n');
fprintf('l = %g, u = %g\n', l, u);
fprintf('coeffs = [%g, %g]\n', coeffs(1), coeffs(2));

% Call computeApproxError
[coeffs_out, d] = layer.computeApproxError(l, u, coeffs);

fprintf('Result: d = %g\n', d);

% Test case: m <= 0
coeffs = [-0.1, 0];  % m = -0.1 < 0

fprintf('\nTesting LeakyReLU m <= 0 case:\n');
fprintf('l = %g, u = %g\n', l, u);
fprintf('coeffs = [%g, %g]\n', coeffs(1), coeffs(2));

% Call computeApproxError
[coeffs_out, d] = layer.computeApproxError(l, u, coeffs);

fprintf('Result: d = %g\n', d);

% Test case: m = alpha
coeffs = [0.01, 0];  % m = 0.01 = alpha

fprintf('\nTesting LeakyReLU m = alpha case:\n');
fprintf('l = %g, u = %g\n', l, u);
fprintf('coeffs = [%g, %g]\n', coeffs(1), coeffs(2));

% Call computeApproxError
[coeffs_out, d] = layer.computeApproxError(l, u, coeffs);

fprintf('Result: d = %g\n', d);
