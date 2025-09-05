% Test ReLU polynomial approximation
clear; clc;

% Create ReLU layer
layer = nnReLULayer();

% Test polynomial approximation
l = -1;
u = 1;
order = 1;
method = "regression";

fprintf('Testing ReLU polynomial approximation:\n');
fprintf('l = %g, u = %g, order = %d, method = %s\n', l, u, order, method);

% Call computeApproxPoly
[coeffs, d] = layer.computeApproxPoly(l, u, order, method);

fprintf('coeffs = [%g, %g]\n', coeffs(1), coeffs(2));
fprintf('d = %g\n', d);

% Test evaluation at some points
x_test = linspace(l, u, 10);
y_true = layer.f(x_test);
y_approx = polyval(coeffs, x_test);

fprintf('\nEvaluation results:\n');
for i = 1:length(x_test)
    fprintf('x = %g: true = %g, approx = %g\n', x_test(i), y_true(i), y_approx(i));
end
