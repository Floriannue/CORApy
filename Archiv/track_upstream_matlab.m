% track_upstream_matlab - Track upstream computations in MATLAB

addpath('cora_matlab/models/auxiliary/jetEngine');

dim_x = 2;
params.tFinal = 8;
params.R0 = zonotope([[1;1],0.1*diag(ones(dim_x,1))]);
params.U = zonotope(0);

options.alg = 'lin-adaptive';
options.trackUpstream = true;
options.trackOptimaldeltat = true;

% Initialize global logs
global upstreamLogGlobal optimaldeltatLogGlobal RtpTrackingGlobal;
upstreamLogGlobal = [];
optimaldeltatLogGlobal = [];
RtpTrackingGlobal = [];

sys = nonlinearSys(@jetEngine,dim_x,1);

fprintf('Running MATLAB with upstream tracking...\n');
[R,~,opt] = reach(sys,params,options);

% Extract logs
if ~isempty(upstreamLogGlobal)
    upstreamLog = upstreamLogGlobal;
    fprintf('\nCaptured %d upstream computation entries\n', length(upstreamLog));
    
    % Extract optimaldeltat log
    optimaldeltatLog = [];
    if ~isempty(optimaldeltatLogGlobal)
        optimaldeltatLog = optimaldeltatLogGlobal;
        fprintf('Captured %d optimaldeltat entries\n', length(optimaldeltatLog));
    end
    
    % Extract Rtp tracking from global variable
    global RtpTrackingGlobal;
    Rtp_tracking = struct();
    if exist('RtpTrackingGlobal', 'var') && ~isempty(RtpTrackingGlobal)
        Rtp_tracking = RtpTrackingGlobal;
        fprintf('Captured Rtp tracking for %d steps\n', length(fieldnames(Rtp_tracking)));
    else
        fprintf('No Rtp tracking found\n');
    end
    
    % Save to file (upstreamLog + optimaldeltatLog + Rtp_tracking)
    if ~isempty(optimaldeltatLog) && ~isempty(fieldnames(Rtp_tracking))
        save('upstream_matlab_log.mat', 'upstreamLog', 'optimaldeltatLog', 'Rtp_tracking', '-v7');
    elseif ~isempty(optimaldeltatLog)
        save('upstream_matlab_log.mat', 'upstreamLog', 'optimaldeltatLog', '-v7');
    elseif ~isempty(fieldnames(Rtp_tracking))
        save('upstream_matlab_log.mat', 'upstreamLog', 'Rtp_tracking', '-v7');
    else
        save('upstream_matlab_log.mat', 'upstreamLog', '-v7');
    end
    fprintf('Saved to upstream_matlab_log.mat\n');
    
    % Show first few entries
    fprintf('\nFirst 3 upstream entries:\n');
    for i = 1:min(3, length(upstreamLog))
        fprintf('\nEntry %d (Step %d, Run %d):\n', i, upstreamLog(i).step, upstreamLog(i).run);
        if isfield(upstreamLog(i), 'Z_before_quadmap') && ~isempty(upstreamLog(i).Z_before_quadmap) && isstruct(upstreamLog(i).Z_before_quadmap) && isfield(upstreamLog(i).Z_before_quadmap, 'radius_max')
            fprintf('  Z before quadMap: radius_max=%.6e\n', upstreamLog(i).Z_before_quadmap.radius_max);
        end
        if isfield(upstreamLog(i), 'errorSec_before_combine') && ~isempty(upstreamLog(i).errorSec_before_combine) && isstruct(upstreamLog(i).errorSec_before_combine) && isfield(upstreamLog(i).errorSec_before_combine, 'radius_max')
            fprintf('  errorSec before combine: radius_max=%.6e\n', upstreamLog(i).errorSec_before_combine.radius_max);
        end
        if isfield(upstreamLog(i), 'VerrorDyn_before_reduce') && ~isempty(upstreamLog(i).VerrorDyn_before_reduce) && isstruct(upstreamLog(i).VerrorDyn_before_reduce) && isfield(upstreamLog(i).VerrorDyn_before_reduce, 'radius_max')
            fprintf('  VerrorDyn before reduce: radius_max=%.6e\n', upstreamLog(i).VerrorDyn_before_reduce.radius_max);
        end
        if isfield(upstreamLog(i), 'VerrorDyn_after_reduce') && ~isempty(upstreamLog(i).VerrorDyn_after_reduce) && isstruct(upstreamLog(i).VerrorDyn_after_reduce) && isfield(upstreamLog(i).VerrorDyn_after_reduce, 'radius_max')
            fprintf('  VerrorDyn after reduce: radius_max=%.6e\n', upstreamLog(i).VerrorDyn_after_reduce.radius_max);
        end
        if isfield(upstreamLog(i), 'Rerror_before_optimaldeltat') && ~isempty(upstreamLog(i).Rerror_before_optimaldeltat) && isstruct(upstreamLog(i).Rerror_before_optimaldeltat) && isfield(upstreamLog(i).Rerror_before_optimaldeltat, 'rerr1')
            fprintf('  Rerror before optimaldeltat: rerr1=%.6e\n', upstreamLog(i).Rerror_before_optimaldeltat.rerr1);
        end
    end
else
    fprintf('No upstream log found\n');
end
