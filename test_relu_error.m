% Test ReLU computeApproxError edge cases
clear; clc;

% Create ReLU layer
layer = nnReLULayer();

% Test case: m >= 1
l = -1;
u = 1;
coeffs = [1.0, 0];  % m = 1.0

fprintf('Testing ReLU m >= 1 case:\n');
fprintf('l = %g, u = %g\n', l, u);
fprintf('coeffs = [%g, %g]\n', coeffs(1), coeffs(2));

% Call computeApproxError
[coeffs_out, d] = layer.computeApproxError(l, u, coeffs);

fprintf('Result: d = %g\n', d);

% Test case: m <= 0
coeffs = [-0.1, 0];  % m = -0.1 < 0

fprintf('\nTesting ReLU m <= 0 case:\n');
fprintf('l = %g, u = %g\n', l, u);
fprintf('coeffs = [%g, %g]\n', coeffs(1), coeffs(2));

% Call computeApproxError
[coeffs_out, d] = layer.computeApproxError(l, u, coeffs);

fprintf('Result: d = %g\n', d);
