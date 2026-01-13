% debug_matlab_initReach_inputDependence.m
% MATLAB debug script to generate I/O pairs for initReach_inputDependence
% 
% This script tests the initReach_inputDependence function for linearParamSys
% and outputs all intermediate values for comparison with Python.

clear; close all; clc;

% Add CORA to path if needed
% addpath(genpath('path/to/cora'));

fprintf('=== Testing initReach_inputDependence ===\n\n');

% Test Case 1: Simple 2D system with constant parameters
fprintf('--- Test Case 1: Simple 2D system ---\n');
A = [0 1; -1 -0.5];
B = [0; 1];
c = [0; 0];
sys1 = linParamSys(A, B, c, 'constParam');

% Initial set
Rinit1 = zonotope([0; 0], [0.1 0; 0 0.1]);

% Parameters
params1 = struct();
params1.Uconst = zonotope([0; 0], [0.05 0; 0 0.05]);
params1.uTrans = [0.1; 0];

% Options
options1 = struct();
options1.timeStep = 0.1;
options1.taylorTerms = 4;
options1.reductionTechnique = 'girard';
options1.zonotopeOrder = 10;
options1.compTimePoint = true;
options1.intermediateTerms = 2;

try
    [sys1_out, Rfirst1, options1_out] = initReach_inputDependence(sys1, Rinit1, params1, options1);
    
    fprintf('sys1_out.taylorTerms: %d\n', sys1_out.taylorTerms);
    fprintf('sys1_out.stepSize: %.10f\n', sys1_out.stepSize);
    fprintf('Rfirst1.ti center:\n');
    disp(center(Rfirst1.ti));
    fprintf('Rfirst1.ti generators:\n');
    disp(generators(Rfirst1.ti));
    if ~isempty(Rfirst1.tp)
        fprintf('Rfirst1.tp center:\n');
        disp(center(Rfirst1.tp));
        fprintf('Rfirst1.tp generators:\n');
        disp(generators(Rfirst1.tp));
    end
    
    fprintf('\nTest Case 1: SUCCESS\n\n');
catch ME
    fprintf('Test Case 1: FAILED - %s\n\n', ME.message);
end

% Test Case 2: System with interval matrix A
fprintf('--- Test Case 2: System with interval matrix A ---\n');
A_int = intervalMatrix([0 1; -1 -0.5], [0.1 0; 0 0.1]);
B2 = [0; 1];
c2 = [0; 0];
sys2 = linParamSys(A_int, B2, c2, 'constParam');

Rinit2 = zonotope([0; 0], [0.1 0; 0 0.1]);
params2 = struct();
params2.Uconst = zonotope([0; 0], [0.05 0; 0 0.05]);
params2.uTrans = [0.1; 0];
options2 = struct();
options2.timeStep = 0.05;
options2.taylorTerms = 3;
options2.reductionTechnique = 'girard';
options2.zonotopeOrder = 10;
options2.compTimePoint = true;
options2.intermediateTerms = 2;

try
    [sys2_out, Rfirst2, options2_out] = initReach_inputDependence(sys2, Rinit2, params2, options2);
    
    fprintf('sys2_out.taylorTerms: %d\n', sys2_out.taylorTerms);
    fprintf('sys2_out.stepSize: %.10f\n', sys2_out.stepSize);
    fprintf('Rfirst2.ti center:\n');
    disp(center(Rfirst2.ti));
    fprintf('Rfirst2.ti generators:\n');
    disp(generators(Rfirst2.ti));
    
    fprintf('\nTest Case 2: SUCCESS\n\n');
catch ME
    fprintf('Test Case 2: FAILED - %s\n\n', ME.message);
end

fprintf('=== Debug script complete ===\n');
