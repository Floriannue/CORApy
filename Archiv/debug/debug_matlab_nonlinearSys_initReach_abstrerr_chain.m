% debug_matlab_nonlinearSys_initReach_abstrerr_chain.m
% MATLAB debug script to generate I/O pairs for the dependency chain:
%   initReach -> linReach -> priv_abstrerr_lin/priv_abstrerr_poly -> priv_precompStatError
%
% Based on: cora_matlab/unitTests/contDynamics/nonlinearSys/test_nonlinearSys_initReach.m
% and how linReach calls these functions internally
%
% Authors: Generated for Python test verification
% Written: 2025

clear; close all; clc;

% Add CORA to path
addpath(genpath('cora_matlab'));

fprintf('=== MATLAB Debug: nonlinearSys initReach -> abstrerr chain ===\n\n');

% Use the 6D tank example (same as MATLAB test)
fprintf('Setting up 6D tank system...\n');
dim_x = 6;
params.R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2*eye(dim_x));
params.U = zonotope(0, 0.005);
params.tFinal = 4;

% Reachability settings (same as MATLAB test)
options.timeStep = 4;
options.taylorTerms = 4;
options.zonotopeOrder = 50;
options.alg = 'lin';
options.tensorOrder = 2;

% Create system
tank = nonlinearSys(@tank6Eq);

% Options check (required, sets maxError and other defaults)
[params, options] = validateOptions(tank, params, options, 'FunctionName', 'reach');

% Compute derivatives (required before initReach)
fprintf('Computing derivatives...\n');
derivatives(tank, options);

% Obtain factors for reachability analysis
for i = 1:(options.taylorTerms+1)
    options.factor(i) = (options.timeStep^i) / factorial(i);
end

% Test 1: initReach with alg='lin' (calls priv_abstrerr_lin via linReach)
fprintf('\n--- Test 1: initReach with alg=''lin'' (calls priv_abstrerr_lin) ---\n');
options.alg = 'lin';
options.tensorOrder = 2;

Rfirst_lin = initReach(tank, params.R0, params, options);

% Extract results (same as MATLAB test)
IH_tp_lin = interval(Rfirst_lin.tp{1}.set);
IH_ti_lin = interval(Rfirst_lin.ti{1});
linErrors_lin = Rfirst_lin.tp{1}.error;

fprintf('IH_tp (lin):\n');
fprintf('  inf: [%s]\n', mat2str(IH_tp_lin.inf, 15));
fprintf('  sup: [%s]\n', mat2str(IH_tp_lin.sup, 15));
fprintf('IH_ti (lin):\n');
fprintf('  inf: [%s]\n', mat2str(IH_ti_lin.inf, 15));
fprintf('  sup: [%s]\n', mat2str(IH_ti_lin.sup, 15));
fprintf('linErrors (lin):\n');
fprintf('  [%s]\n', mat2str(linErrors_lin, 15));

% Test 2: initReach with alg='poly' (calls priv_abstrerr_poly via linReach)
fprintf('\n--- Test 2: initReach with alg=''poly'' (calls priv_abstrerr_poly) ---\n');
% Reset options - poly algorithm needs intermediateOrder
% Note: tensorOrder >= 3 is needed for priv_precompStatError to be called
% (see linReach.m line 80: if options.tensorOrder > 2)
options.alg = 'poly';
options.tensorOrder = 3;  % Use 3 so priv_precompStatError is called
options.intermediateOrder = 10;  % Required for poly algorithm
options.errorOrder = 10;  % Also needed
% Re-validate to ensure all options are set correctly
[params, options] = validateOptions(tank, params, options, 'FunctionName', 'reach');
% Recompute derivatives with tensorOrder = 3 (needed for hessian)
fprintf('Recomputing derivatives with tensorOrder=3...\n');
derivatives(tank, options);
% Recompute factors
for i = 1:(options.taylorTerms+1)
    options.factor(i) = (options.timeStep^i) / factorial(i);
end

Rfirst_poly = initReach(tank, params.R0, params, options);

IH_tp_poly = interval(Rfirst_poly.tp{1}.set);
IH_ti_poly = interval(Rfirst_poly.ti{1});
linErrors_poly = Rfirst_poly.tp{1}.error;

fprintf('IH_tp (poly):\n');
fprintf('  inf: [%s]\n', mat2str(IH_tp_poly.inf, 15));
fprintf('  sup: [%s]\n', mat2str(IH_tp_poly.sup, 15));
fprintf('IH_ti (poly):\n');
fprintf('  inf: [%s]\n', mat2str(IH_ti_poly.inf, 15));
fprintf('  sup: [%s]\n', mat2str(IH_ti_poly.sup, 15));
fprintf('linErrors (poly):\n');
fprintf('  [%s]\n', mat2str(linErrors_poly, 15));

% Test 3: Test linReach directly to see intermediate values
fprintf('\n--- Test 3: linReach directly (to see priv_abstrerr_lin call) ---\n');
options.alg = 'lin';
options.tensorOrder = 2;
[params, options] = validateOptions(tank, params, options, 'FunctionName', 'reach');
for i = 1:(options.taylorTerms+1)
    options.factor(i) = (options.timeStep^i) / factorial(i);
end

% Prepare Rstart structure (as expected by linReach)
Rstart.set = params.R0;
Rstart.error = zeros(dim_x, 1);

[Rti_lin, Rtp_lin, dimForSplit_lin, options_lin] = linReach(tank, Rstart, params, options);

fprintf('linReach results (alg=''lin''):\n');
fprintf('  Rti type: %s\n', class(Rti_lin));
fprintf('  Rtp.set type: %s\n', class(Rtp_lin.set));
fprintf('  Rtp.error: [%s]\n', mat2str(Rtp_lin.error, 15));
fprintf('  dimForSplit: %s\n', mat2str(dimForSplit_lin));

% Test 4: Test linReach with alg='poly' to see priv_precompStatError and priv_abstrerr_poly
fprintf('\n--- Test 4: linReach with alg=''poly'' (calls priv_precompStatError and priv_abstrerr_poly) ---\n');
options.alg = 'poly';
options.tensorOrder = 3;  % Use 3 so priv_precompStatError is called
options.intermediateOrder = 10;  % Required for poly algorithm
options.errorOrder = 10;  % Also needed
[params, options] = validateOptions(tank, params, options, 'FunctionName', 'reach');
% Derivatives should already be computed from Test 2, but ensure they're available
if ~isfield(tank, 'hessian')
    fprintf('Recomputing derivatives with tensorOrder=3...\n');
    derivatives(tank, options);
end
for i = 1:(options.taylorTerms+1)
    options.factor(i) = (options.timeStep^i) / factorial(i);
end

Rstart.set = params.R0;
Rstart.error = zeros(dim_x, 1);

[Rti_poly, Rtp_poly, dimForSplit_poly, options_poly] = linReach(tank, Rstart, params, options);

fprintf('linReach results (alg=''poly''):\n');
fprintf('  Rti type: %s\n', class(Rti_poly));
fprintf('  Rtp.set type: %s\n', class(Rtp_poly.set));
fprintf('  Rtp.error: [%s]\n', mat2str(Rtp_poly.error, 15));
fprintf('  dimForSplit: %s\n', mat2str(dimForSplit_poly));

fprintf('\n=== Debug script completed ===\n');
fprintf('\nNote: priv_abstrerr_lin, priv_abstrerr_poly, and priv_precompStatError\n');
fprintf('are called internally by linReach. The abstraction errors are stored\n');
fprintf('in Rfirst.tp{1}.error for both alg=''lin'' and alg=''poly''.\n');
