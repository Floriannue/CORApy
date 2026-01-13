% debug_matlab_expmMixed.m
% MATLAB debug script to generate I/O pairs for expmMixed
% 
% This script tests the expmMixed function and outputs all intermediate
% values for comparison with Python.

clear; close all; clc;

% Add CORA to path if needed
% addpath(genpath('path/to/cora'));

fprintf('=== Testing expmMixed ===\n\n');

% Test Case 1: Simple 2x2 matrix zonotope
fprintf('--- Test Case 1: Simple 2x2 matrix zonotope ---\n');
C = [0 1; -1 -0.5];
G = zeros(2, 2, 2);
G(:, :, 1) = [0.1 0; 0 0.1];
G(:, :, 2) = [0 0.05; 0.05 0];
matZ = matZonotope(C, G);

r = 0.1;
intermediateOrder = 2;
maxOrder = 4;

try
    [eZ, eI, zPow, iPow, E] = expmMixed(matZ, r, intermediateOrder, maxOrder);
    
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
intermediateOrder2 = 2;
maxOrder2 = 3;

try
    [eZ2, eI2, zPow2, iPow2, E2] = expmMixed(matZ2, r2, intermediateOrder2, maxOrder2);
    
    fprintf('eZ2 center:\n');
    disp(eZ2.C);
    fprintf('eI2 center:\n');
    disp(center(eI2.int));
    
    fprintf('\nTest Case 2: SUCCESS\n\n');
catch ME
    fprintf('Test Case 2: FAILED - %s\n\n', ME.message);
end

fprintf('=== Debug script complete ===\n');
