% debug_matlab_expmOneParam.m
% MATLAB debug script to generate I/O pairs for expmOneParam
% 
% This script tests the expmOneParam function and outputs all intermediate
% values for comparison with Python.

clear; close all; clc;

% Add CORA to path if needed
% addpath(genpath('path/to/cora'));

fprintf('=== Testing expmOneParam ===\n\n');

% Test Case 1: Simple 2x2 matrix zonotope with one generator
fprintf('--- Test Case 1: Simple 2x2 matrix zonotope ---\n');
C = [0 1; -1 -0.5];
G = zeros(2, 2, 1);
G(:, :, 1) = [0.1 0; 0 0.1];
matZ = matZonotope(C, G);

r = 0.1;
maxOrder = 4;
params = struct();
params.Uconst = zonotope([0; 0], [0.05 0; 0 0.05]);
params.uTrans = [0.1; 0];

try
    [eZ, eI, zPow, iPow, E, RconstInput] = expmOneParam(matZ, r, maxOrder, params);
    
    fprintf('eZ center:\n');
    disp(eZ.C);
    fprintf('eZ generators:\n');
    disp(eZ.G);
    fprintf('eI center:\n');
    disp(center(eI.int));
    fprintf('eI radius:\n');
    disp(rad(eI.int));
    fprintf('E center:\n');
    disp(center(E.int));
    fprintf('E radius:\n');
    disp(rad(E.int));
    fprintf('RconstInput center:\n');
    disp(center(RconstInput));
    fprintf('RconstInput generators:\n');
    disp(RconstInput.G);
    fprintf('Number of zPow: %d\n', length(zPow));
    fprintf('Number of iPow: %d\n', length(iPow));
    
    fprintf('\nTest Case 1: SUCCESS\n\n');
catch ME
    fprintf('Test Case 1: FAILED - %s\n\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s:%d in %s\n', ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
    end
end

% Test Case 2: 3x3 matrix zonotope
fprintf('--- Test Case 2: 3x3 matrix zonotope ---\n');
C2 = [0 1 0; -1 -0.5 0; 0 0 -1];
G2 = zeros(3, 3, 1);
G2(:, :, 1) = [0.1 0 0; 0 0.1 0; 0 0 0.1];
matZ2 = matZonotope(C2, G2);

r2 = 0.05;
maxOrder2 = 3;
params2 = struct();
params2.Uconst = zonotope([0; 0; 0], [0.05 0 0; 0 0.05 0; 0 0 0.05]);
params2.uTrans = [0.1; 0; 0];

try
    [eZ2, eI2, zPow2, iPow2, E2, RconstInput2] = expmOneParam(matZ2, r2, maxOrder2, params2);
    
    fprintf('eZ2 center:\n');
    disp(eZ2.C);
    fprintf('eZ2 generators shape: %s\n', mat2str(size(eZ2.G)));
    fprintf('eI2 center:\n');
    disp(center(eI2.int));
    
    fprintf('\nTest Case 2: SUCCESS\n\n');
catch ME
    fprintf('Test Case 2: FAILED - %s\n\n', ME.message);
end

fprintf('=== Debug script complete ===\n');
