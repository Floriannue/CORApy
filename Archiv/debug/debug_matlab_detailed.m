% Detailed debug script to trace computeApproxPoly execution
% This will help us understand exactly why d is not empty

% Create ReLU layer
layer = nnReLULayer();

% Test parameters
l = -1;
u = 1;
order = 2;
method = 'regression';

fprintf('=== MATLAB Detailed Debug ===\n');
fprintf('Testing computeApproxPoly with l=%.1f, u=%.1f, order=%d, method=%s\n', l, u, order, method);

% Test the computeApproxPoly method step by step
fprintf('\n1. Calling computeApproxPoly...\n');

% Manually replicate the computeApproxPoly logic
fprintf('2. Input validation...\n');
if l == u
    fprintf('   l == u case (should not happen)\n');
    return;
elseif l > u
    fprintf('   l > u case (should not happen)\n');
    return;
end

fprintf('3. Initialize coeffs and d...\n');
coeffs = [];
d = [];
fprintf('   coeffs is empty: %s\n', string(isempty(coeffs)));
fprintf('   d is empty: %s\n', string(isempty(d)));

fprintf('4. Computing approximation polynomial...\n');
numPoints = 10*(order+1);
fprintf('   numPoints: %d\n', numPoints);

if strcmp(method, 'regression')
    fprintf('   Using regression method\n');
    x = linspace(l, u, numPoints);
    y = layer.f(x);
    fprintf('   x: [%.3f, %.3f, %.3f, %.3f, %.3f]... (first 5 points)\n', x(1:5));
    fprintf('   y: [%.3f, %.3f, %.3f, %.3f, %.3f]... (first 5 points)\n', y(1:5));
    
    % compute polynomial that best fits the activation function
    coeffs = nnHelper.leastSquarePolyFunc(x, y, order);
    fprintf('   coeffs after leastSquarePolyFunc: [%.8f, %.8f, %.8f]\n', coeffs(1), coeffs(2), coeffs(3));
    fprintf('   d after regression: is empty = %s\n', string(isempty(d)));
end

fprintf('5. Checking if custom method is called...\n');
try
    [coeffs_custom, d_custom] = computeApproxPolyCustom(layer, l, u, order, method);
    fprintf('   Custom method called successfully\n');
    fprintf('   coeffs_custom is empty: %s\n', string(isempty(coeffs_custom)));
    fprintf('   d_custom is empty: %s\n', string(isempty(d_custom)));
    if ~isempty(coeffs_custom)
        coeffs = coeffs_custom;
    end
    if ~isempty(d_custom)
        d = d_custom;
    end
catch ME
    fprintf('   Custom method not available or failed: %s\n', ME.message);
end

fprintf('6. Final coefficient and error processing...\n');
fprintf('   coeffs is empty: %s\n', string(isempty(coeffs)));
fprintf('   d is empty: %s\n', string(isempty(d)));

if isempty(coeffs)
    fprintf('   ERROR: Unable to determine coeffs\n');
elseif isempty(d)
    fprintf('   Computing approx error...\n');
    [coeffs, d] = computeApproxError(layer, l, u, coeffs);
    fprintf('   coeffs after computeApproxError: [%.8f, %.8f, %.8f]\n', coeffs(1), coeffs(2), coeffs(3));
    fprintf('   d after computeApproxError: %.8f\n', d);
else
    fprintf('   d is not empty, skipping computeApproxError\n');
    fprintf('   final coeffs: [%.8f, %.8f, %.8f]\n', coeffs(1), coeffs(2), coeffs(3));
    fprintf('   final d: %.8f\n', d);
end

fprintf('\n7. Comparing with direct call...\n');
[coeffs_direct, d_direct] = layer.computeApproxPoly(l, u, order, method);
fprintf('   Direct call coeffs: [%.8f, %.8f, %.8f]\n', coeffs_direct(1), coeffs_direct(2), coeffs_direct(3));
fprintf('   Direct call d: %.8f\n', d_direct);
