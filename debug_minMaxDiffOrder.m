% Debug script to test minMaxDiffOrder function
% This will help us understand why Python and MATLAB give different results

% Create ReLU layer
layer = nnReLULayer();

% Test parameters
l = -1;
u = 1;
order = 2;
method = 'regression';

% Get initial coefficients (before error computation)
[coeffs_initial, d_initial] = layer.computeApproxPoly(l, u, order, method);

% Test minMaxDiffOrder function directly
[df_l, df_u] = layer.getDerBounds(l, u);

fprintf('Testing minMaxDiffOrder function:\n');
fprintf('  l: %.1f, u: %.1f\n', l, u);
fprintf('  df_l: %.8f, df_u: %.8f\n', df_l, df_u);
fprintf('  coeffs_initial: [%.8f, %.8f, %.8f]\n', coeffs_initial(1), coeffs_initial(2), coeffs_initial(3));

% Call minMaxDiffOrder
[diffl, diffu] = nnHelper.minMaxDiffOrder(coeffs_initial, l, u, layer.f, df_l, df_u);

fprintf('  diffl: %.8f, diffu: %.8f\n', diffl, diffu);

% Compute final values
diffc = (diffl + diffu) / 2;
coeffs_final = coeffs_initial;
coeffs_final(end) = coeffs_final(end) + diffc;
d_final = diffu - diffc;

fprintf('  diffc: %.8f\n', diffc);
fprintf('  coeffs_final: [%.8f, %.8f, %.8f]\n', coeffs_final(1), coeffs_final(2), coeffs_final(3));
fprintf('  d_final: %.8f\n', d_final);

% Compare with direct call
[coeffs_direct, d_direct] = layer.computeApproxPoly(l, u, order, method);
fprintf('\nDirect call results:\n');
fprintf('  coeffs: [%.8f, %.8f, %.8f]\n', coeffs_direct(1), coeffs_direct(2), coeffs_direct(3));
fprintf('  d: %.8f\n', d_direct);
