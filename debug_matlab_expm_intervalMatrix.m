% debug_matlab_expm_intervalMatrix.m
% MATLAB debug script to generate I/O pairs for intervalMatrix expm
% 
% This script tests the expm function for intervalMatrix and outputs all
% intermediate values for comparison with Python.

clear; close all; clc;

% Add CORA to path if needed
% addpath(genpath('path/to/cora'));

fprintf('=== Testing intervalMatrix expm ===\n\n');

% Test Case 1: Simple 2x2 interval matrix
fprintf('--- Test Case 1: Simple 2x2 interval matrix ---\n');
C = [0 1; -1 -0.5];
D = [0.1 0; 0 0.1];
intMat = intervalMatrix(C, D);

maxOrder = 4;

try
    eI = expm(intMat, maxOrder);
    
    fprintf('eI center:\n');
    disp(center(eI.int));
    fprintf('eI radius:\n');
    disp(rad(eI.int));
    
    fprintf('\nTest Case 1: SUCCESS\n\n');
catch ME
    fprintf('Test Case 1: FAILED - %s\n\n', ME.message);
end

% Test Case 2: With r and maxOrder
fprintf('--- Test Case 2: With r and maxOrder ---\n');
r = 0.1;
maxOrder2 = 3;

try
    [eI2, iPow2, E2] = expm(intMat, r, maxOrder2);
    
    fprintf('eI2 center:\n');
    disp(center(eI2.int));
    fprintf('eI2 radius:\n');
    disp(rad(eI2.int));
    fprintf('E2 center:\n');
    disp(center(E2.int));
    fprintf('E2 radius:\n');
    disp(rad(E2.int));
    fprintf('Number of iPow2: %d\n', length(iPow2));
    
    fprintf('\nTest Case 2: SUCCESS\n\n');
catch ME
    fprintf('Test Case 2: FAILED - %s\n\n', ME.message);
end

% Test Case 3: 3x3 interval matrix
fprintf('--- Test Case 3: 3x3 interval matrix ---\n');
C3 = [0 1 0; -1 -0.5 0; 0 0 -1];
D3 = [0.1 0 0; 0 0.1 0; 0 0 0.1];
intMat3 = intervalMatrix(C3, D3);

r3 = 0.05;
maxOrder3 = 3;

try
    [eI3, iPow3, E3] = expm(intMat3, r3, maxOrder3);
    
    fprintf('eI3 center:\n');
    disp(center(eI3.int));
    fprintf('eI3 radius:\n');
    disp(rad(eI3.int));
    
    fprintf('\nTest Case 3: SUCCESS\n\n');
catch ME
    fprintf('Test Case 3: FAILED - %s\n\n', ME.message);
end

fprintf('=== Debug script complete ===\n');
