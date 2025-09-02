% Test LeakyReLU computeApproxError edge cases
layer = nnLeakyReLULayer(0.01);

% Test with m >= 1 (slope >= 1)
l = -1; u = 1;
coeffs = [1.0, 0.0];  % m = 1.0
[new_coeffs, d] = layer.computeApproxError(l, u, coeffs);
fprintf('m >= 1 case: d = %.6f\n', d);

% Test with m <= 0 (slope <= 0)
coeffs = [-0.1, 0.0];  % m = -0.1 < 0
[new_coeffs, d] = layer.computeApproxError(l, u, coeffs);
fprintf('m <= 0 case: d = %.6f\n', d);

% Test with m = alpha (should be exact for negative region)
coeffs = [0.01, 0.0];  % m = 0.01 = alpha
[new_coeffs, d] = layer.computeApproxError(l, u, coeffs);
fprintf('m = alpha case: d = %.6f\n', d);
