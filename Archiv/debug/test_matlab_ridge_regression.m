% Test MATLAB leastSquareRidgePolyFunc behavior
clear; clc;

fprintf('Testing MATLAB leastSquareRidgePolyFunc:\n\n');

% Test 1: Linear function y = 2x + 1
fprintf('=== Test 1: Linear function y = 2x + 1 ===\n');
x = [0, 1, 2, 3];
y = 2 * x + 1;
order = 1;

% Test with default lambda (0.001)
coeffs_default = nnHelper.leastSquareRidgePolyFunc(x, y, order);
fprintf('Default lambda (0.001): coeffs = [%.6f, %.6f]\n', coeffs_default(1), coeffs_default(2));

% Test with lambda = 0 (should be exact)
coeffs_exact = nnHelper.leastSquareRidgePolyFunc(x, y, order, 0);
fprintf('Lambda = 0 (exact): coeffs = [%.6f, %.6f]\n', coeffs_exact(1), coeffs_exact(2));

% Test with lambda = 0.01
coeffs_reg = nnHelper.leastSquareRidgePolyFunc(x, y, order, 0.01);
fprintf('Lambda = 0.01: coeffs = [%.6f, %.6f]\n', coeffs_reg(1), coeffs_reg(2));

fprintf('\n');

% Test 2: Quadratic function y = x² + 2x + 1
fprintf('=== Test 2: Quadratic function y = x² + 2x + 1 ===\n');
x = [0, 1, 2, 3, 4];
y = x.^2 + 2*x + 1;
order = 2;

% Test with default lambda (0.001)
coeffs_default = nnHelper.leastSquareRidgePolyFunc(x, y, order);
fprintf('Default lambda (0.001): coeffs = [%.6f, %.6f, %.6f]\n', coeffs_default(1), coeffs_default(2), coeffs_default(3));

% Test with lambda = 0 (should be exact)
coeffs_exact = nnHelper.leastSquareRidgePolyFunc(x, y, order, 0);
fprintf('Lambda = 0 (exact): coeffs = [%.6f, %.6f, %.6f]\n', coeffs_exact(1), coeffs_exact(2), coeffs_exact(3));

% Test with lambda = 0.01
coeffs_reg = nnHelper.leastSquareRidgePolyFunc(x, y, order, 0.01);
fprintf('Lambda = 0.01: coeffs = [%.6f, %.6f, %.6f]\n', coeffs_reg(1), coeffs_reg(2), coeffs_reg(3));

fprintf('\n');

% Test 3: Edge cases
fprintf('=== Test 3: Edge cases ===\n');

% Single point
x = [1];
y = [5];
order = 0;
coeffs = nnHelper.leastSquareRidgePolyFunc(x, y, order);
fprintf('Single point (order 0): coeffs = [%.6f]\n', coeffs(1));

% Two points
x = [1, 2];
y = [3, 5];
order = 1;
coeffs = nnHelper.leastSquareRidgePolyFunc(x, y, order);
fprintf('Two points (order 1): coeffs = [%.6f, %.6f]\n', coeffs(1), coeffs(2));

fprintf('\n');

% Test 4: Error cases (MATLAB behavior)
fprintf('=== Test 4: Error cases (MATLAB behavior) ===\n');

try
    % Negative order
    coeffs = nnHelper.leastSquareRidgePolyFunc([1, 2, 3], [2, 4, 6], -1);
    fprintf('Negative order: SUCCESS (unexpected!)\n');
catch ME
    fprintf('Negative order: ERROR - %s\n', ME.message);
end

try
    % Order >= number of points
    coeffs = nnHelper.leastSquareRidgePolyFunc([1, 2, 3], [2, 4, 6], 3);
    fprintf('Order >= points: SUCCESS (unexpected!)\n');
catch ME
    fprintf('Order >= points: ERROR - %s\n', ME.message);
end

try
    % Mismatched lengths
    coeffs = nnHelper.leastSquareRidgePolyFunc([1, 2, 3], [2, 4], 1);
    fprintf('Mismatched lengths: SUCCESS (unexpected!)\n');
catch ME
    fprintf('Mismatched lengths: ERROR - %s\n', ME.message);
end

try
    % Empty arrays
    coeffs = nnHelper.leastSquareRidgePolyFunc([], [], 1);
    fprintf('Empty arrays: SUCCESS (unexpected!)\n');
catch ME
    fprintf('Empty arrays: ERROR - %s\n', ME.message);
end

try
    % Negative lambda
    coeffs = nnHelper.leastSquareRidgePolyFunc([1, 2, 3], [2, 4, 6], 1, -0.1);
    fprintf('Negative lambda: SUCCESS (unexpected!)\n');
catch ME
    fprintf('Negative lambda: ERROR - %s\n', ME.message);
end

fprintf('\nDone.\n');
