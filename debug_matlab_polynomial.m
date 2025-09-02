% Debug MATLAB minMaxDiffPoly step by step
clear; clc;

% Test case
coeffs1 = [-0.1, 0];  % p1(x) = -0.1*x + 0
coeffs2 = [0.01, 0];  % p2(x) = 0.01*x + 0
l = -1;
u = 0;

fprintf('MATLAB minMaxDiffPoly debug:\n');
fprintf('coeffs1 = [%g, %g]\n', coeffs1(1), coeffs1(2));
fprintf('coeffs2 = [%g, %g]\n', coeffs2(1), coeffs2(2));
fprintf('l = %g, u = %g\n', l, u);

% MATLAB implementation step by step
% compute difference polynomial: p_1(x) - p_2(x)
p = zeros(1,max(length(coeffs1),length(coeffs2)));
p(end-length(coeffs1)+1:end) = coeffs1;
p(end-length(coeffs2)+1:end) = p(end-length(coeffs2)+1:end)-coeffs2;

fprintf('\nStep by step:\n');
fprintf('max_len = %d\n', max(length(coeffs1),length(coeffs2)));
fprintf('p (after padding coeffs1) = [%g, %g]\n', p(1), p(2));
fprintf('p (after subtracting coeffs2) = [%g, %g]\n', p(1), p(2));

% determine extreme points
dp = fpolyder(p);
fprintf('dp (derivative) = %g\n', dp);

dp_roots = roots(dp);
fprintf('dp_roots = %g\n', dp_roots);

dp_roots = dp_roots(imag(dp_roots) == 0); % filter imaginary roots
fprintf('dp_roots (real only) = %g\n', dp_roots);

dp_roots = dp_roots(l < dp_roots & dp_roots < u);
fprintf('dp_roots (in domain) = %g\n', dp_roots);

extrema = [l, dp_roots', u]; % extrema or boundary
fprintf('extrema = [%g, %g]\n', extrema(1), extrema(2));

diff = polyval(p, extrema);
fprintf('diff values = [%g, %g]\n', diff(1), diff(2));

% compute final approx error
diffl = min(diff);
diffu = max(diff);

fprintf('\nFinal result:\n');
fprintf('diffl = %g, diffu = %g\n', diffl, diffu);
