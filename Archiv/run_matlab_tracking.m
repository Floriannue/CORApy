% run_matlab_tracking - Run MATLAB tracking and save results
% This script runs the tracking and saves to upstream_matlab_log.mat

fprintf('Running MATLAB upstream tracking...\n');

% Run the tracking script
track_upstream_matlab;

fprintf('MATLAB tracking complete!\n');
fprintf('Results saved to upstream_matlab_log.mat\n');
