% debug_or.m
addpath(genpath('D:\Bachelorarbeit\Translate_Cora\cora_matlab'));
sdpsettings('solver', 'SDPT3', 'verbose', 1);

% Define E1 and E0 as in test_ellipsoid_or.py::test_ellipsoid_or_contains_random_points_zero_rank
E1_matlab = ellipsoid([5.43878115, 12.49771836; 12.49771836, 29.66621173], [-0.74450683; 3.58006475]);
E0_matlab = ellipsoid([0.0, 0.0; 0.0, 0.0], [1.09869336; -1.98843878]);

% Compute the union
Eres_0_matlab = or(E1_matlab, E0_matlab);

fprintf('MATLAB Eres_0.Q:\n'); disp(Eres_0_matlab.Q);
fprintf('MATLAB Eres_0.q:\n'); disp(Eres_0_matlab.q);

% Generate random points from E1 (as in Python test)
Y_0_E1_matlab = randPoint(E1_matlab, 2, 'standard');

% Get the point from E0
Y_0_E0_matlab = E0_matlab.q;

% Concatenate points
Y_0_matlab = [Y_0_E1_matlab, Y_0_E0_matlab];

% Check containment in MATLAB
contains_results_matlab = contains(Eres_0_matlab, Y_0_matlab);
fprintf('MATLAB contains_results:\n'); disp(contains_results_matlab);

% For verification with Python, you might also want to see the ellipsoidNorm values in MATLAB
for k = 1:size(Y_0_matlab,2)
    scaling_matlab = ellipsoidNorm(Eres_0_matlab, Y_0_matlab(:,k));
    fprintf('MATLAB Point %d scaling: %f\n', k-1, scaling_matlab);
end 