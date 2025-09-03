% Debug MATLAB polynomial representation
clear; clc;

% Test what [1, 2] represents in MATLAB
p = [1, 2];
fprintf('Polynomial [1, 2] represents: ');
fprintf('%g + %g*x\n', p(2), p(1));

% Test evaluation
x = 0;
y = polyval(p, x);
fprintf('At x=0: %g\n', y);

x = 1;
y = polyval(p, x);
fprintf('At x=1: %g\n', y);

% Test derivative
dp = fpolyder(p);
fprintf('Derivative [%s] represents: ', num2str(dp));
if length(dp) == 1
    fprintf('%g\n', dp(1));
else
    fprintf('%g + %g*x\n', dp(2), dp(1));
end

% Test derivative evaluation at x=0
y_der = polyval(dp, 0);
fprintf('Derivative at x=0: %g\n', y_der);

% Test with [2, 3, 4, 5]
fprintf('\n--- Testing [2, 3, 4, 5] ---\n');
p2 = [2, 3, 4, 5];
fprintf('Polynomial [2, 3, 4, 5] represents: ');
fprintf('%g + %g*x + %g*x^2 + %g*x^3\n', p2(4), p2(3), p2(2), p2(1));

dp2 = fpolyder(p2);
fprintf('Derivative [%s]\n', num2str(dp2));

% Test derivative evaluation at x=0
y_der2 = polyval(dp2, 0);
fprintf('Derivative at x=0: %g\n', y_der2);

fprintf('\nMATLAB polynomial debug completed!\n');