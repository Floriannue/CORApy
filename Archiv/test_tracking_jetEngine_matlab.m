% test_tracking_jetEngine_matlab.m
% Test script for MATLAB intermediate value tracking with jetEngine model

clear; close all; clc;

% Add CORA to path if needed
% addpath(genpath('cora_matlab'));

% Create jetEngine nonlinear system
nlnsys = nonlinearSys(@jetEngine,2,1);

% Set up parameters with longer time horizon to capture error growth
params.tStart = 0;
params.tFinal = 5.0;  % Longer time to capture error_adm_horizon growth
params.R0 = zonotope([1; 1], 0.1*eye(2));
params.U = zonotope(zeros(1,1), []);  % Empty input set

% Set up options with tracking enabled
options.alg = 'lin-adaptive';
options.traceIntermediateValues = true;  % Enable tracking
options.progress = true;
options.progressInterval = 10;  % Less frequent updates for longer run
options.verbose = 0;  % Disable verbose logging

% Set default options that may be required
% These are typically set by CORA's default options, but we set them explicitly
if ~isfield(options, 'isHessianConst')
    options.isHessianConst = false;
end
if ~isfield(options, 'hessianCheck')
    options.hessianCheck = false;
end

% Run reachability analysis
fprintf('=== Testing Intermediate Value Tracking with jetEngine (MATLAB) ===\n');
fprintf('Running reach_adaptive with tracking enabled...\n');

tic;
[timeInt, timePoint, res, tVec, options] = reach_adaptive(nlnsys, params, options);
tComp = toc;

fprintf('[OK] reach_adaptive completed in %.2f seconds\n', tComp);

% Check for trace files
trace_files = dir('intermediate_values_step*_inner_loop.txt');
if ~isempty(trace_files)
    fprintf('\n[OK] Found %d trace file(s):\n', length(trace_files));
    for i = 1:min(5, length(trace_files))
        file_info = dir(trace_files(i).name);
        fprintf('  - %s (%d bytes)\n', trace_files(i).name, file_info.bytes);
        
        % Read first few lines to show sample
        fid = fopen(trace_files(i).name, 'r');
        if fid ~= -1
            fprintf('    First iteration sample:\n');
            line_count = 0;
            while ~feof(fid) && line_count < 10
                line = fgetl(fid);
                if ~isempty(line) && ~startsWith(line, '===')
                    fprintf('      %s\n', line);
                    line_count = line_count + 1;
                end
            end
            fclose(fid);
        end
    end
    if length(trace_files) > 5
        fprintf('  ... and %d more\n', length(trace_files) - 5);
    end
else
    fprintf('\n[WARNING] No trace files found. Tracking may not be working.\n');
end

fprintf('\n=== Test Complete ===\n');
fprintf('\nTo compare with Python:\n');
fprintf('1. Run Python with equivalent tracking enabled\n');
fprintf('2. Use: python compare_intermediate_values.py <matlab_file> <python_file> [tolerance]\n');
