% Debug script to test MATLAB findBernsteinPoly function
% Compare with Python implementation

% Test with constant function
f = @(x) 5.0;
l = 0; u = 1; n = 3;

coeffs_const = findBernsteinPoly(f, l, u, n);
fprintf('MATLAB Constant function coeffs: [');
fprintf('%.6f ', coeffs_const);
fprintf(']\n');

% Test polynomial evaluation at endpoints
x0 = 0; x1 = 1;
val0 = polyval(coeffs_const, x0);
val1 = polyval(coeffs_const, x1);
fprintf('At x=0: %.6f\n', val0);
fprintf('At x=1: %.6f\n', val1);

% Test with linear function
g = @(x) 2*x + 1;
coeffs_linear = findBernsteinPoly(g, l, u, n);
fprintf('\nMATLAB Linear function coeffs: [');
fprintf('%.6f ', coeffs_linear);
fprintf(']\n');

val0_linear = polyval(coeffs_linear, x0);
val1_linear = polyval(coeffs_linear, x1);
fprintf('At x=0: %.6f (expected: %.6f)\n', val0_linear, g(x0));
fprintf('At x=1: %.6f (expected: %.6f)\n', val1_linear, g(x1));

% Test with quadratic function
h = @(x) x^2;
coeffs_quad = findBernsteinPoly(h, l, u, n);
fprintf('\nMATLAB Quadratic function coeffs: [');
fprintf('%.6f ', coeffs_quad);
fprintf(']\n');

% Test accuracy at several points
test_points = linspace(l, u, 5);
fprintf('\nQuadratic function accuracy test:\n');
for i = 1:length(test_points)
    x = test_points(i);
    bernstein_val = polyval(coeffs_quad, x);
    original_val = h(x);
    error = abs(bernstein_val - original_val);
    fprintf('x=%.3f: Bernstein=%.6f, Original=%.6f, Error=%.6f\n', ...
            x, bernstein_val, original_val, error);
end

% Test with higher order
n_high = 5;
coeffs_quad_high = findBernsteinPoly(h, l, u, n_high);
fprintf('\nMATLAB Quadratic function (n=5) coeffs: [');
fprintf('%.6f ', coeffs_quad_high);
fprintf(']\n');

% Test accuracy with higher order
fprintf('\nQuadratic function accuracy test (n=5):\n');
for i = 1:length(test_points)
    x = test_points(i);
    bernstein_val = polyval(coeffs_quad_high, x);
    original_val = h(x);
    error = abs(bernstein_val - original_val);
    fprintf('x=%.3f: Bernstein=%.6f, Original=%.6f, Error=%.6f\n', ...
            x, bernstein_val, original_val, error);
end
