% Debug script to test MATLAB regression method step by step

% Create ReLU layer
layer = nnReLULayer();

% Test parameters
l = -1;
u = 1;
order = 2;
method = 'regression';

fprintf('MATLAB Regression Debug:\n');
fprintf('  l: %.1f, u: %.1f\n', l, u);
fprintf('  order: %d, method: %s\n', order, method);

% Test the regression method step by step
numPoints = 10 * (order + 1);
x = linspace(l, u, numPoints);
y = layer.f(x);

fprintf('  numPoints: %d\n', numPoints);
fprintf('  x: [%.3f, %.3f, %.3f, %.3f, %.3f]... (first 5 points)\n', x(1), x(2), x(3), x(4), x(5));
fprintf('  y: [%.3f, %.3f, %.3f, %.3f, %.3f]... (first 5 points)\n', y(1), y(2), y(3), y(4), y(5));

% Test leastSquarePolyFunc directly
coeffs_direct = nnHelper.leastSquarePolyFunc(x, y, order);
fprintf('  coeffs_direct: [%.8f, %.8f, %.8f]\n', coeffs_direct(1), coeffs_direct(2), coeffs_direct(3));

% Test computeApproxError directly
[coeffs_error, d_error] = layer.computeApproxError(l, u, coeffs_direct);
fprintf('  coeffs_after_error: [%.8f, %.8f, %.8f]\n', coeffs_error(1), coeffs_error(2), coeffs_error(3));
fprintf('  d_after_error: %.8f\n', d_error);

% Now test the full method
[coeffs_full, d_full] = layer.computeApproxPoly(l, u, order, method);
fprintf('\nFull method results:\n');
fprintf('  coeffs: [%.8f, %.8f, %.8f]\n', coeffs_full(1), coeffs_full(2), coeffs_full(3));
fprintf('  d: %.8f\n', d_full);

% Check if d is empty in the method
fprintf('\nChecking if d is empty in the method:\n');
fprintf('  d_direct is empty: %s\n', mat2str(isempty(d_error)));
fprintf('  d_full is empty: %s\n', mat2str(isempty(d_full)));
