% Debug MATLAB computeApproxError for m <= 0 case
clear; clc;

% Create LeakyReLU layer
layer = nnLeakyReLULayer(0.01);

% Test case: m <= 0
l = -1;
u = 1;
coeffs = [-0.1, 0];  % m = -0.1 < 0

fprintf('Testing m <= 0 case:\n');
fprintf('l = %g, u = %g\n', l, u);
fprintf('coeffs = [%g, %g]\n', coeffs(1), coeffs(2));

% Call computeApproxError
[coeffs_out, d] = layer.computeApproxError(l, u, coeffs);

fprintf('Result: d = %g\n', d);

% Let's debug step by step
fprintf('\nStep-by-step debug:\n');

% x < 0: p(x) - alpha*x
coeffs_alpha = [layer.alpha, 0];
fprintf('coeffs_alpha = [%g, %g]\n', coeffs_alpha(1), coeffs_alpha(2));
[diffl1, diffu1] = minMaxDiffPoly(coeffs, coeffs_alpha, l, 0);
fprintf('diffl1 = %g, diffu1 = %g\n', diffl1, diffu1);

% x > 0: p(x) - 1*x  
coeffs_one = [1, 0];
fprintf('coeffs_one = [%g, %g]\n', coeffs_one(1), coeffs_one(2));
[diffl2, diffu2] = minMaxDiffPoly(coeffs, coeffs_one, 0, u);
fprintf('diffl2 = %g, diffu2 = %g\n', diffl2, diffu2);

% Final computation
diffl = min(diffl1, diffl2);
diffu = max(diffu1, diffu2);
diffc = (diffu + diffl) / 2;
d_calc = diffu - diffc;

fprintf('diffl = %g, diffu = %g\n', diffl, diffu);
fprintf('diffc = %g, d_calc = %g\n', diffc, d_calc);
