% test_matlab_comparison.m
% MATLAB test for comparing error_adm_horizon growth with Python
% Uses tensor order 2 to avoid hessian function requirements

clear; close all; clc;

% Add CORA to path if needed
% addpath(genpath('cora_matlab'));

% Create vanDerPol nonlinear system (simpler than jetEngine)
sys = nonlinearSys(@vanderPolEq,2,1);

% Set up parameters
params.tStart = 0;
params.tFinal = 5.0;  % Longer time to capture error_adm_horizon growth
params.R0 = zonotope([1; 1], 0.1*eye(2));
params.U = zonotope(zeros(1,1), []);  % Empty input set

% Get default options
try
    options = CORAoptions();
catch
    options = struct();
end

% Set algorithm - use tensor order 2 to avoid hessian issues
options.alg = 'lin-adaptive';
options.tensorOrder = 2;  % Use tensor order 2 instead of 3

% Enable tracking
options.traceIntermediateValues = true;
options.progress = true;
options.progressInterval = 10;
options.verbose = 0;

% Set required fields
if ~isfield(options, 'isHessianConst')
    options.isHessianConst = false;
end
if ~isfield(options, 'hessianCheck')
    options.hessianCheck = false;
end
if ~isfield(options, 'thirdOrderTensorempty')
    options.thirdOrderTensorempty = true;  % No third order tensor needed
end

% Run reachability analysis
fprintf('=== Testing Intermediate Value Tracking with vanDerPol (MATLAB) ===\n');
fprintf('Running reach_adaptive with tracking enabled...\n');
fprintf('Time horizon: %.1f to %.1f\n', params.tStart, params.tFinal);
fprintf('Tensor order: %d\n', options.tensorOrder);
fprintf('This may take a while...\n\n');

tic;
try
    [timeInt, timePoint, res, tVec, options] = reach_adaptive(sys, params, options);
    tComp = toc;
    
    fprintf('[OK] reach_adaptive completed in %.2f seconds\n', tComp);
    
    % Check for trace files
    trace_files = dir('intermediate_values_step*_inner_loop.txt');
    if ~isempty(trace_files)
        fprintf('\n[OK] Found %d trace file(s):\n', length(trace_files));
        
        % Sort by step number
        [~, idx] = sort([trace_files.datenum]);
        trace_files = trace_files(idx);
        
        for i = 1:min(10, length(trace_files))
            fprintf('  - %s (%d bytes)\n', trace_files(i).name, trace_files(i).bytes);
        end
        if length(trace_files) > 10
            fprintf('  ... and %d more\n', length(trace_files) - 10);
        end
        
        % Analyze error_adm_horizon growth
        fprintf('\n=== Analyzing error_adm_horizon Growth ===\n');
        error_adm_horizon_values = [];
        for i = 1:length(trace_files)
            fname = trace_files(i).name;
            fid = fopen(fname, 'r');
            if fid ~= -1
                content = fread(fid, '*char')';
                fclose(fid);
                
                % Extract step number
                step_match = regexp(fname, 'step(\d+)', 'tokens');
                if ~isempty(step_match)
                    step_num = str2double(step_match{1}{1});
                    
                    % Extract initial error_adm_horizon
                    match = regexp(content, 'Initial error_adm_horizon:\s*\[\[([\d.e+-]+)\]', 'tokens');
                    if ~isempty(match)
                        val = str2double(match{1}{1});
                        error_adm_horizon_values = [error_adm_horizon_values; step_num, val];
                    end
                end
            end
        end
        
        if ~isempty(error_adm_horizon_values)
            error_adm_horizon_values = sortrows(error_adm_horizon_values, 1);
            fprintf('Found error_adm_horizon values for %d steps\n', size(error_adm_horizon_values, 1));
            fprintf('Step | error_adm_horizon_max\n');
            fprintf('------------------------------\n');
            for i = 1:min(20, size(error_adm_horizon_values, 1))
                fprintf('%4d | %.6e\n', error_adm_horizon_values(i,1), error_adm_horizon_values(i,2));
            end
            if size(error_adm_horizon_values, 1) > 20
                fprintf('... and %d more\n', size(error_adm_horizon_values, 1) - 20);
            end
            
            % Growth analysis
            if size(error_adm_horizon_values, 1) > 1
                first_val = error_adm_horizon_values(1, 2);
                last_val = error_adm_horizon_values(end, 2);
                growth_factor = last_val / first_val;
                fprintf('\nGrowth analysis:\n');
                fprintf('  First step: %.6e\n', first_val);
                fprintf('  Last step:  %.6e\n', last_val);
                fprintf('  Growth factor: %.2fx\n', growth_factor);
                if growth_factor > 10
                    fprintf('  [WARNING] Significant growth detected!\n');
                end
            end
        end
    else
        fprintf('\n[WARNING] No trace files found. Tracking may not be working.\n');
    end
    
    fprintf('\n=== Test Complete ===\n');
    fprintf('\nTo compare with Python:\n');
    fprintf('1. Run Python with equivalent tracking enabled\n');
    fprintf('2. Use: python compare_intermediate_values.py <matlab_file> <python_file> [tolerance]\n');
    
catch ME
    fprintf('\n[ERROR] Test failed: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).file, ME.stack(i).line);
    end
    rethrow(ME);
end
