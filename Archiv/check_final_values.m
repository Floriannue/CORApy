% Check final values from MATLAB run
clear; close all; clc;

% Load or check the results
fprintf('=== Checking MATLAB Final Values ===\n');

% Check trace files to see how many steps were completed
trace_files = dir('intermediate_values_step*_inner_loop_matlab.txt');
if ~isempty(trace_files)
    fprintf('Found %d MATLAB trace files\n', length(trace_files));
    
    % Get step numbers
    step_nums = [];
    for i = 1:length(trace_files)
        fname = trace_files(i).name;
        step_match = regexp(fname, 'step(\d+)', 'tokens');
        if ~isempty(step_match)
            step_nums = [step_nums, str2double(step_match{1}{1})];
        end
    end
    
    if ~isempty(step_nums)
        fprintf('Step numbers range: %d to %d\n', min(step_nums), max(step_nums));
        fprintf('Total unique steps: %d\n', length(unique(step_nums)));
    end
    
    % Check last trace file
    [~, idx] = sort(step_nums);
    last_file = trace_files(idx(end)).name;
    fprintf('\nLast trace file: %s\n', last_file);
    
    % Read last file
    fid = fopen(last_file, 'r');
    if fid ~= -1
        content = fread(fid, '*char')';
        fclose(fid);
        
        % Extract final error_adm_horizon
        match = regexp(content, 'Final error_adm_horizon:\s*\[\[([\d.e+-]+)\]', 'tokens');
        if ~isempty(match)
            fprintf('Final error_adm_horizon: %s\n', match{1}{1});
        end
        
        % Extract step number from content
        step_match = regexp(content, 'Step (\d+)', 'tokens');
        if ~isempty(step_match)
            fprintf('Step number in file: %s\n', step_match{1}{1});
        end
    end
end

% Check if we can determine final time from the reach_adaptive output
fprintf('\n=== Expected Final Time ===\n');
fprintf('tFinal should be: 2.0\n');
fprintf('If MATLAB stopped early, check aux_checkForAbortion logic\n');
