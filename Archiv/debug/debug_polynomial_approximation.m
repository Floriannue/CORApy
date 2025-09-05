% Debug script to compare MATLAB and Python polynomial approximation
% for ReLU layer with order 2 regression

% Create ReLU layer
layer = nnReLULayer();

% Test parameters
l = -1;
u = 1;
order = 2;
method = 'regression';

% Call the method
[coeffs, d] = layer.computeApproxPoly(l, u, order, method);

% Display results
fprintf('MATLAB Results:\n');
fprintf('  coeffs: [%.8f, %.8f, %.8f]\n', coeffs(1), coeffs(2), coeffs(3));
fprintf('  d: %.8f\n', d);

% Test evaluation at some points
x_test = linspace(l, u, 5);
y_true = layer.f(x_test);
y_approx = polyval(coeffs, x_test);

fprintf('\nEvaluation test:\n');
fprintf('  x_test: [%.1f, %.1f, %.1f, %.1f, %.1f]\n', x_test);
fprintf('  y_true: [%.1f, %.1f, %.1f, %.1f, %.1f]\n', y_true);
fprintf('  y_approx: [%.8f, %.8f, %.8f, %.8f, %.8f]\n', y_approx);

% Let's also check what the parent class does
fprintf('\nParent class method (nnActivationLayer):\n');
[coeffs_parent, d_parent] = layer.computeApproxPoly(l, u, order, method);
fprintf('  coeffs: [%.8f, %.8f, %.8f]\n', coeffs_parent(1), coeffs_parent(2), coeffs_parent(3));
fprintf('  d: %.8f\n', d_parent);

% Check if ReLU has its own implementation
fprintf('\nChecking method resolution:\n');
methods = methods(layer);
if any(contains({methods.Name}, 'computeApproxPoly'))
    fprintf('  ReLU layer has its own computeApproxPoly method\n');
else
    fprintf('  ReLU layer uses parent class computeApproxPoly method\n');
end
