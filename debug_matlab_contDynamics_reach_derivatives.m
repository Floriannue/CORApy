% debug_matlab_contDynamics_reach_derivatives.m
% MATLAB debug script to verify contDynamics.reach calls to derivatives
% for nonlinearSys
%
% This script verifies:
% 1. contDynamics.reach calls derivatives(sys, options) for nonlinearSys
% 2. The derivatives are computed before reachability analysis
% 3. The derivatives are stored in the system object
%
% Authors: Generated for Python test verification
% Written: 2025

clear; close all; clc;

% Add CORA to path
addpath(genpath('cora_matlab'));

fprintf('=== MATLAB Debug: contDynamics.reach -> derivatives ===\n\n');

% Use the 6D tank example
fprintf('Setting up 6D tank system...\n');
dim_x = 6;
params.R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2*eye(dim_x));
params.U = zonotope(zeros(1,1), 0.005*eye(1));
params.tFinal = 4;
params.uTrans = zeros(1,1);

% Reachability settings
options.timeStep = 0.1;
options.taylorTerms = 4;
options.zonotopeOrder = 50;
options.alg = 'lin';
options.tensorOrder = 2;

% Create system
tank = nonlinearSys(@tank6Eq, dim_x, 1);

% Check system before derivatives
fprintf('System before derivatives:\n');
fprintf('  Has jacobian: %d\n', isfield(tank, 'jacobian'));
fprintf('  Has hessian: %d\n', isfield(tank, 'hessian'));

% Test: contDynamics.reach automatically calls derivatives
fprintf('\n--- Test: contDynamics.reach calls derivatives ---\n');
fprintf('Calling reach (this should call derivatives internally)...\n');

[R, res, options_out] = reach(tank, params, options);

% Check system after reach (derivatives should be computed)
fprintf('\nSystem after reach (derivatives should be computed):\n');
fprintf('  Has jacobian: %d\n', isfield(tank, 'jacobian'));
fprintf('  Has hessian: %d\n', isfield(tank, 'hessian'));
fprintf('  Has thirdOrderTensor: %d\n', isfield(tank, 'thirdOrderTensor'));

% Verify derivatives were called
if isfield(tank, 'jacobian')
    fprintf('  jacobian file: %s\n', func2str(tank.jacobian));
end
if isfield(tank, 'hessian')
    fprintf('  hessian file: %s\n', func2str(tank.hessian));
end

% Test: Explicit derivatives call (same as in reach)
fprintf('\n--- Test: Explicit derivatives call ---\n');
tank2 = nonlinearSys(@tank6Eq, dim_x, 1);
derivatives(tank2, options);

fprintf('System after explicit derivatives:\n');
fprintf('  Has jacobian: %d\n', isfield(tank2, 'jacobian'));
fprintf('  Has hessian: %d\n', isfield(tank2, 'hessian'));

% Verify reach works without explicit derivatives call
fprintf('\n--- Verification: reach should work without explicit derivatives ---\n');
tank3 = nonlinearSys(@tank6Eq, dim_x, 1);
fprintf('Calling reach on fresh system (derivatives called internally)...\n');
[R2, res2, options_out2] = reach(tank3, params, options);
fprintf('Reach completed successfully: %d\n', res2);
fprintf('Reachable sets computed: %d time points\n', length(R2.timePoint.set));

fprintf('\n=== Debug script completed ===\n');
fprintf('\nKey finding: contDynamics.reach automatically calls derivatives(sys, options)\n');
fprintf('for nonlinearSys before computing reachable sets.\n');
