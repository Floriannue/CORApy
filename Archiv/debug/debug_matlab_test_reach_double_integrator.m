% Debug script to check dimensions and values for test_reach_double_integrator
% Compare against Python implementation

clear; close all; clc;

% Double integrator system
A = zeros(2, 2);
B = [1; 1];

sys = linearSys(A, B);

% Parameters
params.tFinal = 1.0;
params.R0 = zonotope([1; 1], 0.1*eye(2));
params.U = zonotope(1, 0.1);

% Options
options.timeStep = 0.04;
options.taylorTerms = 4;
options.zonotopeOrder = 10;
options.linAlg = 'wrapping-free';

% Compute reachable set
R = sys.reach(params, options);

% Check final interval hull
final_set = R.timeInterval.set{end};
IH = interval(final_set);

fprintf('=== Final Set Dimensions ===\n');
fprintf('final_set type: %s\n', class(final_set));
if isa(final_set, 'zonotope')
    fprintf('final_set center shape: %s\n', mat2str(size(final_set.c)));
    fprintf('final_set generators shape: %s\n', mat2str(size(final_set.G)));
end

fprintf('\n=== Interval Hull ===\n');
fprintf('IH type: %s\n', class(IH));
fprintf('IH.inf shape: %s\n', mat2str(size(IH.inf)));
fprintf('IH.sup shape: %s\n', mat2str(size(IH.sup)));
fprintf('IH.inf ndim: %d\n', ndims(IH.inf));
fprintf('IH.sup ndim: %d\n', ndims(IH.sup));

fprintf('\n=== Interval Values ===\n');
fprintf('IH.inf:\n');
disp(IH.inf);
fprintf('IH.sup:\n');
disp(IH.sup);

fprintf('\n=== Infimum and Supremum ===\n');
IH_inf = infimum(IH);
IH_sup = supremum(IH);
fprintf('IH_inf shape: %s\n', mat2str(size(IH_inf)));
fprintf('IH_sup shape: %s\n', mat2str(size(IH_sup)));
fprintf('IH_inf:\n');
disp(IH_inf);
fprintf('IH_sup:\n');
disp(IH_sup);

fprintf('\n=== Expected Values (from test) ===\n');
expected_lower = [1.76, 1.76; 1.76, 1.76];
expected_upper = [2.16, 2.16; 2.16, 2.16];
fprintf('expected_lower shape: %s\n', mat2str(size(expected_lower)));
fprintf('expected_upper shape: %s\n', mat2str(size(expected_upper)));
fprintf('expected_lower:\n');
disp(expected_lower);
fprintf('expected_upper:\n');
disp(expected_upper);

fprintf('\n=== Comparison ===\n');
fprintf('IH_inf matches expected_lower: %d\n', isequal(size(IH_inf), size(expected_lower)));
fprintf('IH_sup matches expected_upper: %d\n', isequal(size(IH_sup), size(expected_upper)));
if isequal(size(IH_inf), size(expected_lower))
    fprintf('IH_inf values match (within tolerance): %d\n', all(abs(IH_inf(:) - expected_lower(:)) < 0.01));
end
if isequal(size(IH_sup), size(expected_upper))
    fprintf('IH_sup values match (within tolerance): %d\n', all(abs(IH_sup(:) - expected_upper(:)) < 0.01));
end
