% Test MATLAB polyval behavior
clear; clc;

% Test polynomial: -0.11*x + 0
p = [-0.11, 0];
x_points = [-1, 0];

fprintf('MATLAB polyval test:\n');
fprintf('Polynomial coefficients: [%g, %g]\n', p(1), p(2));
fprintf('Evaluation points: [%g, %g]\n', x_points(1), x_points(2));

for i = 1:length(x_points)
    x = x_points(i);
    val = polyval(p, x);
    fprintf('polyval([%g, %g], %g) = %g\n', p(1), p(2), x, val);
end

% Manual evaluation to confirm
fprintf('\nManual evaluation:\n');
for i = 1:length(x_points)
    x = x_points(i);
    val = p(1) * x + p(2);  % -0.11*x + 0
    fprintf('p(%g) = %g*x + %g = %g\n', x, p(1), p(2), val);
end
