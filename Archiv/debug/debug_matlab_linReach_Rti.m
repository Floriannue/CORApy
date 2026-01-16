% debug_matlab_linReach_Rti.m
% Debug script to capture Rti values in linReach
clear; close all; clc;

addpath(genpath('cora_matlab'));

fprintf('=== MATLAB Debug: linReach Rti values ===\n\n');

% Use the 6D tank example (same as Python test)
dim_x = 6;
params.R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2*eye(dim_x));
params.U = zonotope(0, 0.005);
params.tFinal = 4;

options.timeStep = 4;
options.taylorTerms = 4;
options.zonotopeOrder = 50;
options.alg = 'lin';
options.tensorOrder = 2;

tank = nonlinearSys(@tank6Eq);
[params, options] = validateOptions(tank, params, options, 'FunctionName', 'reach');

derivatives(tank, options);

for i = 1:(options.taylorTerms+1)
    options.factor(i) = (options.timeStep^i) / factorial(i);
end

Rstart.set = params.R0;
Rstart.error = zeros(dim_x, 1);

% Call linReach
[Rti, Rtp, dimForSplit, options_out] = linReach(tank, Rstart, params, options);

fprintf('After linReach (Rti is translated):\n');
fprintf('Rti center: [%s]\n', mat2str(Rti.center, 15));
Rti_int = interval(Rti);
fprintf('Rti interval: inf=[%s], sup=[%s]\n', mat2str(Rti_int.inf, 15), mat2str(Rti_int.sup, 15));
fprintf('Rtp.error: [%s]\n', mat2str(Rtp.error, 15));

fprintf('\n=== Debug completed ===\n');
